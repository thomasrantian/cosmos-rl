# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for model-config routing at the LLM trainer boundary."""

from __future__ import annotations

import pytest
from transformers import PretrainedConfig

from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.trainer.llm_trainer import llm_trainer as llm_trainer_module
from cosmos_rl.utils import model_config


class _StopAfterConfigLoad(RuntimeError):
    """Stop trainer construction before a model or CUDA state is created."""


class _ConcreteLLMTrainer(llm_trainer_module.LLMTrainer):
    """Make the base trainer constructible without selecting an algorithm."""

    def build_lr_schedulers(self) -> None:
        raise NotImplementedError

    def step_training(self) -> None:
        raise NotImplementedError


def _construct_until_model_build(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_name_or_path: str,
) -> None:
    config = CosmosConfig()
    config.policy.model_name_or_path = model_name_or_path
    config.train.seed = None
    config.train.deterministic = False

    monkeypatch.setattr(
        llm_trainer_module,
        "init_flash_attn_meta",
        lambda *_args, **_kwargs: None,
    )
    # A routing regression should fail immediately instead of exercising the
    # production retry backoff against the deliberately failing HF fallback.
    monkeypatch.setattr(llm_trainer_module.util, "retry", lambda function: function)

    def stop_build_model(_cls: type, build_config: CosmosConfig) -> None:
        assert build_config is config
        raise _StopAfterConfigLoad

    monkeypatch.setattr(
        llm_trainer_module.ModelRegistry,
        "build_model",
        classmethod(stop_build_model),
    )

    with pytest.raises(_StopAfterConfigLoad):
        _ConcreteLLMTrainer(
            config=config,
            parallel_dims=object(),
            train_stream=None,
            data_packer=object(),
            val_data_packer=object(),
        )


def test_llm_trainer_resolves_registered_non_hf_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered locator must not be sent to HuggingFace AutoConfig."""

    locator = "test-model-config://policy-only"
    factory_calls: list[str] = []

    class TestPolicyConfig(PretrainedConfig):
        model_type = "test_policy_only"

    # Keep this test isolated from process-global model-config registrations.
    monkeypatch.setattr(model_config, "_CUSTOM_MODEL_CONFIG_LOADERS", [])
    model_config.register_local_model_config(
        predicate=lambda value: value.startswith("test-model-config://"),
        factory=lambda value: factory_calls.append(value) or TestPolicyConfig(),
    )

    def fail_hf_fallback(*_args: object, **_kwargs: object) -> None:
        pytest.fail("registered model locator reached AutoConfig.from_pretrained")

    monkeypatch.setattr(
        model_config.AutoConfig,
        "from_pretrained",
        fail_hf_fallback,
    )

    _construct_until_model_build(
        monkeypatch,
        model_name_or_path=locator,
    )

    assert factory_calls == [locator]


def test_llm_trainer_preserves_huggingface_config_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal model paths retain the previous trusted AutoConfig behavior."""

    model_name_or_path = "test-org/test-model"
    fallback_calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(model_config, "_CUSTOM_MODEL_CONFIG_LOADERS", [])

    def fake_hf_fallback(
        value: str,
        **kwargs: object,
    ) -> PretrainedConfig:
        fallback_calls.append((value, kwargs))
        return PretrainedConfig()

    monkeypatch.setattr(
        model_config.AutoConfig,
        "from_pretrained",
        fake_hf_fallback,
    )

    _construct_until_model_build(
        monkeypatch,
        model_name_or_path=model_name_or_path,
    )

    assert fallback_calls == [
        (model_name_or_path, {"trust_remote_code": True}),
    ]
