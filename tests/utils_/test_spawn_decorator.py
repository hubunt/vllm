# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for spawn_new_process_for_each_test decorator."""

import pytest

from tests import utils
from tests.utils import spawn_new_process_for_each_test


@spawn_new_process_for_each_test
def test_spawn_decorator_passing():
    """Passing function should complete normally."""
    assert 1 + 1 == 2


@pytest.mark.xfail(raises=RuntimeError, strict=True)
@spawn_new_process_for_each_test
def test_spawn_decorator_failure_is_caught():
    """Failing function should raise RuntimeError, never silently pass."""
    raise ValueError("intentional failure")


@spawn_new_process_for_each_test
def test_spawn_decorator_skip():
    """pytest.skip inside subprocess should propagate correctly."""
    pytest.skip("intentional skip")


@spawn_new_process_for_each_test
@pytest.mark.parametrize("x,y,expected", [(1, 2, 3), (0, 0, 0)])
def test_spawn_decorator_parametrized(x, y, expected):
    """Args and kwargs must be forwarded correctly to subprocess."""
    assert x + y == expected


def test_create_new_process_uses_spawn_after_cuda_init(monkeypatch):
    """Forking after parent CUDA init makes child CUDA use fail."""
    selected_methods: list[str] = []

    def _make_decorator(method: str):
        def decorator(func):
            def wrapper(*args, **kwargs):
                selected_methods.append(method)
                return func(*args, **kwargs)

            return wrapper

        return decorator

    monkeypatch.setattr(utils.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(utils.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(utils, "xpu_is_initialized", lambda: False)
    monkeypatch.setattr(utils, "fork_new_process_for_each_test", _make_decorator("fork"))
    monkeypatch.setattr(
        utils, "spawn_new_process_for_each_test", _make_decorator("spawn")
    )

    cuda_initialized = False
    monkeypatch.setattr(utils, "cuda_is_initialized", lambda: cuda_initialized)

    called = False

    @utils.create_new_process_for_each_test()
    def target():
        nonlocal called
        called = True

    cuda_initialized = True
    target()

    assert called
    assert selected_methods == ["spawn"]
