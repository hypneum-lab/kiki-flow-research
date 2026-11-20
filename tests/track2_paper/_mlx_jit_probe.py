"""Probe whether the MLX JIT (``mx.compile``) works on this host.

mlx-cpu 0.31.1 on Linux fails at the g++ step of ``mx.compile`` with
"redeclaration of C++ built-in type '_Float128'" (GCC 13 toolchain
incompatibility, issue #43). Keying the skip on this probe rather than on
``mx.metal.is_available()`` means the MLX tests come back by themselves
on Linux as soon as a fixed mlx-cpu lands, instead of staying skipped
forever on hosts without Metal. The probe runs once per session.
"""

from __future__ import annotations

import functools

import mlx.core as mx


@functools.lru_cache(maxsize=1)
def mlx_jit_available() -> bool:
    """Return True if a trivial ``mx.compile`` round-trips on this host."""
    try:
        compiled = mx.compile(lambda x: x + 1.0)
        out = compiled(mx.array(1.0))
        mx.eval(out)
        return bool(float(out.item()) == 2.0)
    except Exception:  # noqa: BLE001 - any JIT/toolchain failure means "no JIT"
        return False


MLX_JIT_REASON = (
    "mx.compile JIT unavailable on this host (mlx-cpu 0.31.1 on Linux g++: "
    "_Float128 redeclaration); see #43"
)
