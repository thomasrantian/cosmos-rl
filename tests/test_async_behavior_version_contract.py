# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cosmos_rl.dispatcher.command import RolloutToRolloutBroadcastCommand
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.rollout.worker.asynchronous.rollout_task_scheduler import (
    RolloutTask,
    RolloutTaskScheduler,
)
from cosmos_rl.rollout.worker.rollout_control import (
    DisaggregatedRolloutControlWorker,
    PromptVersionDecision,
    _batch_requested_weight_version,
    prompt_version_decision,
)
from cosmos_rl.rollout.worker.weight_sync import AsyncR2RSyncMode


class _FakeRolloutEngine:
    def __init__(self) -> None:
        self.initialized = False

    def is_engine_initialized(self) -> bool:
        return self.initialized

    def shutdown(self) -> None:
        self.initialized = False

    async def rollout_generation(self, **_kwargs):
        return []


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition was not met")
        time.sleep(0.005)


def test_quiesce_drains_admitted_work_and_blocks_new_tasks() -> None:
    engine = _FakeRolloutEngine()
    live_version = [8]
    starts: list[tuple[int, int]] = []
    release = threading.Semaphore(0)

    def generation_fn(**kwargs):
        payload = kwargs["payloads"][0]
        payload.weight_version = live_version[0]
        starts.append((payload.prompt_idx, payload.weight_version))

        async def generate():
            await asyncio.to_thread(release.acquire)
            return [RolloutResult(completions=["ok"])]

        return generate()

    scheduler = RolloutTaskScheduler(
        rollout_engine=engine,
        data_packer=object(),
        max_concurrent_requests=1,
        check_interval=0.005,
        rollout_generation_fn=generation_fn,
    )
    scheduler.start(lambda value: setattr(value, "initialized", True), True)
    scheduler.put_rollout_batch(
        [
            RolloutTask(idx=0, payload=RLPayload(prompt_idx=0, weight_version=99)),
            RolloutTask(idx=1, payload=RLPayload(prompt_idx=1, weight_version=99)),
        ]
    )
    _wait_until(lambda: starts == [(0, 8)])

    quiesce_errors: list[BaseException] = []

    def quiesce() -> None:
        try:
            scheduler.quiesce_after_drain(timeout=2.0)
        except BaseException as exc:  # pragma: no cover - asserted below
            quiesce_errors.append(exc)

    thread = threading.Thread(target=quiesce)
    thread.start()
    _wait_until(lambda: not scheduler._accepting_tasks)

    with pytest.raises(RuntimeError, match="admission is closed"):
        scheduler.put_rollout(
            RolloutTask(idx=2, payload=RLPayload(prompt_idx=2, weight_version=99))
        )

    release.release()
    _wait_until(lambda: starts == [(0, 8), (1, 8)])
    assert thread.is_alive(), "quiesce returned before queued work drained"
    release.release()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert quiesce_errors == []
    assert scheduler.is_paused()
    assert [item.behavior_weight_version for item in scheduler.get_all()] == [8, 8]

    live_version[0] = 9
    scheduler.resume()
    scheduler.put_rollout(
        RolloutTask(idx=2, payload=RLPayload(prompt_idx=2, weight_version=123))
    )
    _wait_until(lambda: starts[-1] == (2, 9))
    release.release()
    _wait_until(lambda: scheduler.complete_queue.qsize() == 1)
    assert scheduler.get_all()[0].behavior_weight_version == 9
    scheduler.stop()


def test_quiesce_timeout_fails_closed() -> None:
    engine = _FakeRolloutEngine()
    started = threading.Event()
    release = threading.Event()

    async def generation_fn(**_kwargs):
        started.set()
        await asyncio.to_thread(release.wait)
        return [RolloutResult(completions=["ok"])]

    scheduler = RolloutTaskScheduler(
        rollout_engine=engine,
        data_packer=object(),
        max_concurrent_requests=1,
        check_interval=0.005,
        rollout_generation_fn=generation_fn,
    )
    scheduler.start(lambda value: setattr(value, "initialized", True), True)
    scheduler.put_rollout(
        RolloutTask(idx=0, payload=RLPayload(prompt_idx=0, weight_version=0))
    )
    assert started.wait(timeout=1.0)

    with pytest.raises(TimeoutError):
        scheduler.quiesce_after_drain(timeout=0.02)
    assert scheduler.is_paused()
    with pytest.raises(RuntimeError, match="poisoned"):
        scheduler.resume()
    with pytest.raises(RuntimeError, match="poisoned"):
        scheduler.put_rollout(
            RolloutTask(idx=1, payload=RLPayload(prompt_idx=1, weight_version=0))
        )

    release.set()
    scheduler.stop()


def test_quiesce_drains_tasks_queued_during_manual_pause() -> None:
    engine = _FakeRolloutEngine()

    async def generation_fn(**_kwargs):
        return [RolloutResult(completions=["ok"])]

    scheduler = RolloutTaskScheduler(
        rollout_engine=engine,
        data_packer=object(),
        check_interval=0.005,
        rollout_generation_fn=generation_fn,
    )
    scheduler.start(lambda value: setattr(value, "initialized", True), True)
    scheduler.pause()
    scheduler.put_rollout(
        RolloutTask(idx=0, payload=RLPayload(prompt_idx=0, weight_version=8))
    )

    scheduler.quiesce_after_drain(timeout=1.0)

    assert scheduler.is_paused()
    assert len(scheduler.get_all()) == 1
    scheduler.stop()


def test_stopped_scheduler_rejects_new_tasks() -> None:
    engine = _FakeRolloutEngine()
    scheduler = RolloutTaskScheduler(
        rollout_engine=engine,
        data_packer=object(),
    )
    scheduler.start(lambda value: setattr(value, "initialized", True), True)
    scheduler.stop()

    with pytest.raises(RuntimeError, match="admission is closed"):
        scheduler.put_rollout(
            RolloutTask(idx=0, payload=RLPayload(prompt_idx=0, weight_version=0))
        )


def test_backend_failure_is_surfaced_and_poisoned() -> None:
    engine = _FakeRolloutEngine()

    async def generation_fn(**_kwargs):
        raise ValueError("simulator failed")

    scheduler = RolloutTaskScheduler(
        rollout_engine=engine,
        data_packer=object(),
        check_interval=0.005,
        rollout_generation_fn=generation_fn,
    )
    scheduler.start(lambda value: setattr(value, "initialized", True), True)
    scheduler.put_rollout(
        RolloutTask(idx=0, payload=RLPayload(prompt_idx=0, weight_version=0))
    )
    _wait_until(scheduler._poisoned.is_set)

    with pytest.raises(RuntimeError, match="generation failed") as error:
        scheduler.raise_if_failed()
    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == "simulator failed"
    with pytest.raises(RuntimeError, match="poisoned"):
        scheduler.put_rollout(
            RolloutTask(idx=1, payload=RLPayload(prompt_idx=1, weight_version=0))
        )
    scheduler.stop()


def test_empty_backend_result_is_surfaced_and_poisoned() -> None:
    engine = _FakeRolloutEngine()

    scheduler = RolloutTaskScheduler(
        rollout_engine=engine,
        data_packer=object(),
        check_interval=0.005,
    )
    scheduler.start(lambda value: setattr(value, "initialized", True), True)
    scheduler.put_rollout(
        RolloutTask(idx=0, payload=RLPayload(prompt_idx=0, weight_version=0))
    )
    _wait_until(scheduler._poisoned.is_set)

    with pytest.raises(RuntimeError, match="generation failed") as error:
        scheduler.raise_if_failed()
    assert isinstance(error.value.__cause__, RuntimeError)
    assert "empty results" in str(error.value.__cause__)
    scheduler.stop()


@pytest.mark.parametrize(
    ("requested", "live", "allowed", "on_policy", "expected"),
    [
        (8, 8, 0, True, PromptVersionDecision.SERVE),
        (7, 8, 0, True, PromptVersionDecision.DROP),
        (9, 8, 0, True, PromptVersionDecision.WAIT),
        (7, 8, 1, False, PromptVersionDecision.SERVE),
        (9, 8, 1, False, PromptVersionDecision.SERVE),
        (10, 8, 1, False, PromptVersionDecision.WAIT),
    ],
)
def test_prompt_version_decision(
    requested: int,
    live: int,
    allowed: int,
    on_policy: bool,
    expected: PromptVersionDecision,
) -> None:
    assert prompt_version_decision(requested, live, allowed, on_policy) == expected


def test_mixed_requested_versions_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="one requested weight version"):
        _batch_requested_weight_version(
            [RLPayload(weight_version=8), RLPayload(weight_version=9)]
        )


def test_generation_start_stamps_actual_live_version() -> None:
    observed: dict[str, object] = {}
    payload = RLPayload(prompt_idx=0, weight_version=7)

    def rollout_generation(**kwargs):
        observed.update(kwargs)
        return [RolloutResult(completions=["ok"])]

    worker = SimpleNamespace(
        current_weight_version=8,
        rollout=SimpleNamespace(rollout_generation=rollout_generation),
    )
    with patch(
        "cosmos_rl.rollout.worker.rollout_control.get_async_r2r_sync_mode",
        return_value=AsyncR2RSyncMode.DISABLED,
    ):
        result = DisaggregatedRolloutControlWorker._call_rollout_generation(
            worker,
            payloads=[payload],
            stream=None,
            data_packer=None,
            is_validation=False,
        )

    assert len(result) == 1
    assert payload.weight_version == 8
    assert observed["current_weight_version"] == 8


def test_generation_wrapper_preserves_strict_backend_signature() -> None:
    observed: list[int] = []
    payload = RLPayload(prompt_idx=0, weight_version=7)

    def rollout_generation(
        *,
        payloads,
        stream,
        data_packer,
        data_fetcher,
        is_validation,
    ):
        del stream, data_packer, data_fetcher, is_validation
        observed.append(payloads[0].weight_version)
        return [RolloutResult(completions=["ok"])]

    worker = SimpleNamespace(
        current_weight_version=8,
        rollout=SimpleNamespace(rollout_generation=rollout_generation),
    )
    with patch(
        "cosmos_rl.rollout.worker.rollout_control.get_async_r2r_sync_mode",
        return_value=AsyncR2RSyncMode.DISABLED,
    ):
        result = DisaggregatedRolloutControlWorker._call_rollout_generation(
            worker,
            payloads=[payload],
            stream=None,
            data_packer=None,
            data_fetcher=None,
            is_validation=False,
        )

    assert len(result) == 1
    assert observed == [8]


def test_scheduler_freezes_behavior_version_before_backend_await() -> None:
    payload = RLPayload(prompt_idx=3, weight_version=7)

    def generation_fn(**kwargs):
        generated_payload = kwargs["payloads"][0]
        generated_payload.weight_version = 8

        async def generate():
            generated_payload.weight_version = 9
            return [RolloutResult(completions=["ok"])]

        return generate()

    scheduler = RolloutTaskScheduler(
        rollout_engine=_FakeRolloutEngine(),
        data_packer=object(),
        rollout_generation_fn=generation_fn,
    )
    completed = asyncio.run(
        scheduler._generate_single(RolloutTask(idx=3, payload=payload))
    )

    assert completed is not None
    assert completed.behavior_weight_version == 8
    assert completed.payload.weight_version == 9


def test_completion_keeps_generation_version_after_live_advance() -> None:
    enqueued: list[tuple[list[RLPayload], int]] = []
    payload = RLPayload(prompt_idx=0, weight_version=7)
    result = RolloutResult(completions=[object()])
    worker = SimpleNamespace(
        config=SimpleNamespace(
            train=SimpleNamespace(
                non_text=True,
                local_dataset=False,
                train_policy=SimpleNamespace(bypass_reward=False),
            ),
            rollout=SimpleNamespace(
                n_generation=1,
                multi_turn_config=SimpleNamespace(enable=False),
            ),
        ),
        current_weight_version=9,
        should_report=True,
        _report_discarded_samples=lambda _count: None,
        enqueue_teacher_calculation=lambda values: values,
        reward_dispatcher=SimpleNamespace(
            enqueue_rewards_cal=lambda values, _validation, step, **_kwargs: (
                enqueued.append((values, step))
            )
        ),
    )

    DisaggregatedRolloutControlWorker._filter_valid_rollout_results_and_report(
        worker, [result], [payload]
    )

    assert payload.weight_version == 7
    assert len(enqueued) == 1
    assert enqueued[0][1] == 7
    assert enqueued[0][0][0].weight_version == 7


def test_scheduler_completion_detects_version_mutation() -> None:
    payload = RLPayload(prompt_idx=0, weight_version=8)
    completed = SimpleNamespace(
        payload=payload,
        result=RolloutResult(completions=["ok"]),
        behavior_weight_version=7,
    )
    worker = SimpleNamespace(
        scheduler=SimpleNamespace(get_all=lambda: [completed]),
    )

    with pytest.raises(RuntimeError, match="mutated after generation"):
        DisaggregatedRolloutControlWorker._stream_generation_collect_results(worker)


def test_async_main_loop_yields_after_scheduler_pump() -> None:
    waits: list[float] = []

    class _ShutdownSignal:
        stopped = False

        def is_set(self) -> bool:
            return self.stopped

        def wait(self, timeout: float) -> bool:
            waits.append(timeout)
            self.stopped = True
            return True

    worker = SimpleNamespace(
        shutdown_signal=_ShutdownSignal(),
        consume_command=lambda **_kwargs: None,
        validation_flag=SimpleNamespace(is_set=lambda: False),
        _maybe_emit_mainloop_summary=lambda _now: None,
        state=SimpleNamespace(weight_synced=lambda: True),
        report_rollouts=lambda: (None, False, None, True),
        _is_async_rollout=True,
        _single_producer_mode=False,
        config=SimpleNamespace(
            rollout=SimpleNamespace(prefetch_rollout=False),
        ),
        stream_generation_step=lambda: None,
        replica_name="rollout-0",
    )

    with patch(
        "cosmos_rl.rollout.worker.rollout_control.get_async_r2r_sync_mode",
        return_value=AsyncR2RSyncMode.DISABLED,
    ):
        DisaggregatedRolloutControlWorker._main_loop_impl(worker)

    assert waits == [0.001]


def test_async_main_loop_yields_while_weights_are_unsynced() -> None:
    waits: list[float] = []

    class _ShutdownSignal:
        stopped = False

        def is_set(self) -> bool:
            return self.stopped

        def wait(self, timeout: float) -> bool:
            waits.append(timeout)
            self.stopped = True
            return True

    worker = SimpleNamespace(
        shutdown_signal=_ShutdownSignal(),
        consume_command=lambda **_kwargs: None,
        validation_flag=SimpleNamespace(is_set=lambda: False),
        _maybe_emit_mainloop_summary=lambda _now: None,
        state=SimpleNamespace(weight_synced=lambda: False),
        _is_async_rollout=True,
        _single_producer_mode=False,
        config=SimpleNamespace(
            rollout=SimpleNamespace(prefetch_rollout=False),
        ),
        replica_name="rollout-0",
    )

    with patch(
        "cosmos_rl.rollout.worker.rollout_control.get_async_r2r_sync_mode",
        return_value=AsyncR2RSyncMode.DISABLED,
    ):
        DisaggregatedRolloutControlWorker._main_loop_impl(worker)

    assert waits == [0.001]


def test_weight_publish_fences_cuda_before_version_and_resume() -> None:
    events: list[tuple[str, int]] = []
    worker = SimpleNamespace(
        replica_name="rollout-0",
        data_packer=SimpleNamespace(),
        _is_async_rollout=True,
        _quiesce_async_scheduler_for_weight_sync=lambda step: events.append(
            ("quiesce", step)
        ),
        _resume_async_scheduler_after_weight_sync=lambda: events.append(
            ("resume", worker.current_weight_version)
        ),
        state=SimpleNamespace(
            weight_synced=lambda: False,
            set_weight_synced=lambda: events.append(
                ("ready", worker.current_weight_version)
            ),
        ),
        config=SimpleNamespace(
            validation=SimpleNamespace(
                enable=False,
                val_before_train=False,
                freq=100,
            )
        ),
        inference_stream=SimpleNamespace(
            synchronize=lambda: events.append(("fence", worker.current_weight_version))
        ),
        current_weight_version=8,
        validation_flag=threading.Event(),
        shutdown_signal=threading.Event(),
        shutdown_mp_signal=threading.Event(),
        redis_controller=SimpleNamespace(publish_teacher_request=lambda *_args: None),
    )
    command = RolloutToRolloutBroadcastCommand(
        src_replica_name="rollout-0",
        dst_replica_names=["rollout-0"],
        weight_step=9,
        total_steps=10,
        trainable_only=True,
    )

    with (
        patch(
            "cosmos_rl.rollout.worker.rollout_control.get_async_r2r_sync_mode",
            return_value=AsyncR2RSyncMode.DISABLED,
        ),
        patch(
            "cosmos_rl.rollout.worker.rollout_control.get_broadcast_all_params",
            return_value=False,
        ),
    ):
        DisaggregatedRolloutControlWorker.broadcast_to_all_rollout_replica(
            worker, command
        )

    assert worker.current_weight_version == 9
    assert events == [
        ("quiesce", 9),
        ("fence", 8),
        ("ready", 9),
        ("resume", 9),
    ]
