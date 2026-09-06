# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from unittest.mock import patch

import requests
import pytest

from cosmos_rl.dispatcher.api.client import APIClient
from cosmos_rl.utils import constant
from cosmos_rl.utils.network_util import make_request_with_retry


def _client() -> APIClient:
    client = object.__new__(APIClient)
    client.max_retries = constant.COSMOS_HTTP_RETRY_CONFIG.max_retries
    client.get_alternative_urls = lambda _suffix: ["http://controller/unregister"]
    return client


def test_unregister_uses_shutdown_specific_retry_budget():
    client = _client()

    with patch(
        "cosmos_rl.dispatcher.api.client.make_request_with_retry"
    ) as make_request:
        client.unregister("policy-0")

    make_request.assert_called_once()
    args, kwargs = make_request.call_args
    request = args[0]
    assert isinstance(request, partial)
    assert request.func is requests.post
    assert request.keywords["json"] == {"replica_name": "policy-0"}
    assert request.keywords["timeout"] == constant.COSMOS_SHUTDOWN_HTTP_TIMEOUT
    assert kwargs == {
        "max_retries": constant.COSMOS_SHUTDOWN_HTTP_MAX_ATTEMPTS,
        "retries_per_delay": 1,
        "initial_delay": 0.0,
        "max_delay": 0.0,
        "backoff_factor": 1.0,
    }
    assert kwargs["max_retries"] < client.max_retries


def test_unregister_failure_is_best_effort():
    client = _client()

    with patch(
        "cosmos_rl.dispatcher.api.client.make_request_with_retry",
        side_effect=requests.ConnectionError("controller already stopped"),
    ) as make_request:
        result = client.unregister("policy-0")

    assert result is None
    assert make_request.call_count == 1


def test_shutdown_retry_defaults_use_one_bounded_attempt_per_endpoint():
    # Total socket time scales with the number of controller URLs, but each URL
    # is visited only once and each request remains independently bounded.
    assert constant.COSMOS_SHUTDOWN_HTTP_MAX_ATTEMPTS == 1
    assert constant.COSMOS_SHUTDOWN_HTTP_TIMEOUT > 0
    assert constant.COSMOS_SHUTDOWN_HTTP_TIMEOUT <= 2.0


def test_one_attempt_visits_each_alternative_once_without_retry_cycle():
    calls = []

    def fail(url):
        calls.append(url)
        raise requests.ConnectionError(f"unreachable: {url}")

    with (
        patch("cosmos_rl.utils.network_util.time.sleep") as sleep,
        pytest.raises(requests.ConnectionError),
    ):
        make_request_with_retry(
            fail,
            urls=["http://controller-a", "http://controller-b"],
            max_retries=1,
            retries_per_delay=1,
            initial_delay=0.0,
            max_delay=0.0,
            backoff_factor=1.0,
        )

    assert calls == ["http://controller-a", "http://controller-b"]
    sleep.assert_called_once_with(0.0)
