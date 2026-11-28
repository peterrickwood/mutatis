"""Programs compiled here and run in worker processes, or in this process: the same results either way, a deadline that stops
them, errors reported, and linear algebra working in a fresh worker."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.numerics.parallel import Local, ParallelError, Workers, compile_program, usable_cpus


def _step(x, total, k):
    """state (x, total) -> (x + k, total + x), and one extra output x * 2."""
    return x + k, total + jnp.sum(x), x * 2.0


def _program():
    # Programs sent to worker processes are compiled afresh, as compile_program requires (and the samplers do): the tests' own
    # compilation cache (tests/conftest.py) would otherwise hand back a cached program that a worker cannot run.
    return compile_program(_step, np.zeros(3), np.float64(0), np.float64(0), use_cache=False)


def test_workers_and_this_process_run_a_compiled_program_alike():
    states = [[np.arange(3.0), np.float64(0)], [np.ones(3), np.float64(1)]]
    steps = [[[np.float64(k)] for k in range(4)], [[np.float64(-k)] for k in range(4)]]
    with Workers(usable_cpus()[:2]) as workers:
        workers.load("step", _program())
        remote = workers.iterate("step", states, steps, deadline=float("inf"))
    local = Local()
    local.load("step", _program())
    for (state, outputs), start, todo in zip(remote, states, steps):
        [(here_state, here_outputs)] = local.iterate("step", [start], [todo], deadline=float("inf"))
        x, total = jnp.asarray(start[0]), jnp.asarray(start[1])
        for (k,) in todo:
            x, total, extra = _step(x, total, k)
        np.testing.assert_array_equal(state[0], x)
        np.testing.assert_array_equal(state[1], total)
        np.testing.assert_array_equal(here_state[0], x)
        assert len(outputs) == len(here_outputs) == 4
        np.testing.assert_array_equal(outputs[-1][0], extra)
        np.testing.assert_array_equal(here_outputs[-1][0], extra)


def test_a_passed_deadline_stops_after_one_step():
    with Workers(usable_cpus()[:1]) as workers:
        workers.load("step", _program())
        [(state, outputs)] = workers.iterate("step", [[np.zeros(3), np.float64(0)]], [[[np.float64(1)]] * 10], deadline=time.time() - 1)
    assert len(outputs) == 1 and np.allclose(state[0], 1.0)


def test_errors_in_a_worker_are_reported():
    with Workers(usable_cpus()[:1]) as workers:
        with pytest.raises(ParallelError, match="failed"):
            workers.iterate("never loaded", [[np.zeros(3)]], [[[np.float64(1)]]], deadline=float("inf"))


def test_linear_algebra_runs_in_a_fresh_worker():
    """A compiled program that calls LAPACK (Cholesky, LU solve, eigenvalues) crashed a fresh worker before the worker loaded
    LAPACK itself, which JAX otherwise does only when it compiles such a call."""
    def solve(a, b):
        return jnp.linalg.cholesky(a @ a.T + jnp.eye(3)), jnp.linalg.solve(a + 3 * jnp.eye(3), b), jnp.linalg.eigvals(a)

    a, b = np.arange(9.0).reshape(3, 3) / 10, np.ones(3)
    with Workers(usable_cpus()[:1]) as workers:
        workers.load("solve", compile_program(solve, a, b, use_cache=False))
        [(_, [outputs])] = workers.iterate("solve", [[]], [[[a, b]]], deadline=float("inf"))
    for got, expected in zip(outputs, jax.jit(solve)(a, b)):
        np.testing.assert_array_equal(got, np.asarray(expected))


def test_programs_for_workers_are_compiled_afresh_even_when_the_compilation_cache_has_them(tmp_path):
    """A program loaded from JAX's persistent compilation cache, then sent to a worker, failed there ("Function ... not
    found"). Programs for workers are compiled with the cache bypassed, so this must run although the cache holds it."""
    from jax.experimental.compilation_cache import compilation_cache

    before = (jax.config.jax_compilation_cache_dir, jax.config.jax_persistent_cache_min_compile_time_secs)
    def program(x):
        return (jnp.cumsum(jnp.outer(x, x) * 2.0, axis=0),)   # programs return a tuple of arrays

    x = np.arange(4.0)
    try:
        jax.config.update("jax_compilation_cache_dir", str(tmp_path))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        compilation_cache.reset_cache()
        compile_program(program, x)                       # fills the cache
        assert any(tmp_path.iterdir())
        jax.clear_caches()
        compiled = compile_program(program, x, use_cache=False)
        assert jax.config.jax_enable_compilation_cache     # left on
        with Workers(usable_cpus()[:1]) as workers:
            workers.load("p", compiled)
            [(_, [[out]])] = workers.iterate("p", [[]], [[[x]]], deadline=float("inf"))
        np.testing.assert_array_equal(out, np.asarray(program(x)[0]))
    finally:
        jax.config.update("jax_compilation_cache_dir", before[0])
        jax.config.update("jax_persistent_cache_min_compile_time_secs", before[1])
        compilation_cache.reset_cache()
