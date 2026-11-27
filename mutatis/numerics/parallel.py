"""Independent work on several CPUs at once: one process per CPU, each pinned to that CPU, running programs compiled once in
this process and sent to it ready to run, so a worker process neither compiles anything nor needs the Python code that made
the program.

Why separate processes (measured 2026-11-27; decision 42): in one process, independent chains do not spread over the cores
(time per chain-step stayed flat from 1 to 64 chains), and splitting them across several XLA host devices in one process was
slower than one device for a Kalman-filter likelihood, whose many small linear-algebra calls contend between threads (4 chains
on 4 devices: 1.0 s, against 0.17 s for one chain alone). The same program in separate processes pinned to their own CPUs ran
at full speed: 0.17 s alone, 0.17 to 0.33 s with 4 at once, 0.23 to 0.42 s with 8 at once (8 CPUs, 4 physical cores).

Why compile here and send the compiled program: compiling the sampler's program for a forward-looking model took 12 s here,
17 s in a process pinned to one CPU, and 29 s in each of 6 pinned processes compiling at once. Loading the compiled program
(`jax.experimental.serialize_executable`) took 0.5 to 0.8 s, and it ran bit for bit like the original.

`Local` runs the same programs in this process, with the same interface, for when there is one CPU: one code path either way.

A program returns a tuple of arrays and is run as a loop on the host: for each step, outputs = program(*state, *step_inputs);
the first outputs become the new state, the rest are kept. The deadline is checked between steps.
"""

import os
import pickle
import subprocess
import sys
import time
import traceback

import numpy as np


def usable_cpus() -> list[int]:
    """The CPUs this process may run on."""
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def compile_program(fn, *example_args, use_cache: bool = True):
    """`fn` compiled for arguments shaped like the examples (a `jax.stages.Compiled`). With `use_cache`, JAX's persistent
    compilation cache is used when one is configured, so a program compiled by an earlier run is not compiled again. A program
    to be sent to worker processes must be compiled afresh: one loaded from the cache sends incompletely, and a worker running
    it fails ("Function broadcast_multiply_fusion.8 not found"; seen 2026-11-27 with JAX 0.12.1, CPU)."""
    import jax

    if use_cache:
        return jax.jit(fn).lower(*example_args).compile()
    from jax.experimental.compilation_cache import compilation_cache

    enabled = jax.config.jax_enable_compilation_cache
    jax.config.update("jax_enable_compilation_cache", False)
    compilation_cache.reset_cache()          # the cache's on/off state is otherwise read once per process
    try:
        return jax.jit(fn).lower(*example_args).compile()
    finally:
        jax.config.update("jax_enable_compilation_cache", enabled)
        compilation_cache.reset_cache()


class ParallelError(RuntimeError):
    pass


def _iterate(run, state: list, steps: list, deadline: float):
    """Steps of one program from one state, stopping early once `time.time()` passes `deadline`: (final state, the extra
    outputs of each step taken)."""
    n, outputs = len(state), []
    for step in steps:
        result = run(*state, *step)
        state, extra = list(result[:n]), result[n:]
        outputs.append([np.asarray(x) for x in extra])
        if time.time() > deadline:
            break
    return [np.asarray(s) for s in state], outputs


class Local:
    """Runs programs in this process, with the interface of `Workers` (one worker)."""

    sends_programs = False

    def __init__(self):
        self._programs = {}

    def __len__(self) -> int:
        return 1

    def load(self, name, compiled):
        self._programs[name] = compiled

    def loaded(self, name) -> bool:
        return name in self._programs

    def iterate(self, name, states: list, steps: list, deadline: float) -> list:
        return [_iterate(self._programs[name], states[0], steps[0], deadline)]

    def close(self):
        self._programs = {}


class Workers:
    """One process per CPU given. Use as a context manager, or call `close`."""

    sends_programs = True   # programs for them must be compiled afresh (see `compile_program`)

    def __init__(self, cpus: list[int]):
        env = dict(os.environ, JAX_PLATFORMS="cpu")
        launch = ("import os, sys; os.sched_setaffinity(0, {%d}); "
                  "from mutatis.numerics.parallel import _serve; _serve()")
        self.cpus = list(cpus)
        self._processes = [subprocess.Popen([sys.executable, "-c", launch % c], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
                           for c in self.cpus]
        self._loaded = set()

    def __len__(self) -> int:
        return len(self._processes)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _send(self, process, message):
        pickle.dump(message, process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
        process.stdin.flush()

    def _receive(self, process):
        try:
            reply = pickle.load(process.stdout)
        except EOFError:
            raise ParallelError(f"a worker process stopped unexpectedly (exit code {process.poll()})") from None
        if isinstance(reply, tuple) and reply and reply[0] == "error":
            raise ParallelError("a worker process failed:\n" + reply[1])
        return reply

    def load(self, name, compiled):
        """Send a compiled program (from `compile_program`) to every process, to be run under `name`."""
        if name in self._loaded:
            return
        from jax.experimental.serialize_executable import serialize

        blob = pickle.dumps(serialize(compiled), protocol=pickle.HIGHEST_PROTOCOL)
        for p in self._processes:
            self._send(p, ("load", name, blob))
        for p in self._processes:
            self._receive(p)
        self._loaded.add(name)

    def loaded(self, name) -> bool:
        return name in self._loaded

    def iterate(self, name, states: list, steps: list, deadline: float) -> list:
        """Run program `name` in every process at once: process k starts from `states[k]` (a list of arrays) and takes the
        steps `steps[k]` (a list of lists of arrays), stopping early once `time.time()` passes `deadline`. Returns, per
        process, (final state, the extra outputs of each step taken)."""
        for p, state, todo in zip(self._processes, states, steps):
            self._send(p, ("iterate", name, [np.asarray(s) for s in state], todo, deadline))
        return [self._receive(p) for p in self._processes]

    def close(self):
        for p in self._processes:
            if p.poll() is None:
                try:
                    self._send(p, ("stop",))
                    p.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
        for p in self._processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        self._processes = []

    def __del__(self):
        if getattr(self, "_processes", None):
            self.close()


def _serve():
    """A worker process's loop. Messages arrive on stdin and replies leave on a private copy of stdout; anything else printed
    goes to stderr, so it cannot corrupt the replies."""
    channel = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    inbox = sys.stdin.buffer
    import jax
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import deserialize_and_load

    jax.config.update("jax_enable_x64", True)
    # JAX loads its LAPACK routines the first time it compiles a linear-algebra operation. A program loaded already compiled
    # skips that, and calling LAPACK before it is loaded crashes the process (seen 2026-11-27), so load it now.
    jnp.linalg.cholesky(jnp.eye(2)).block_until_ready()
    programs = {}
    while True:
        try:
            message = pickle.load(inbox)
        except EOFError:
            return
        if message[0] == "stop":
            return
        try:
            if message[0] == "load":
                _, name, blob = message
                programs[name] = deserialize_and_load(*pickle.loads(blob))
                reply = "ok"
            else:
                _, name, state, steps, deadline = message
                reply = _iterate(programs[name], state, steps, deadline)
        except Exception:
            reply = ("error", traceback.format_exc())
        pickle.dump(reply, channel, protocol=pickle.HIGHEST_PROTOCOL)
        channel.flush()
