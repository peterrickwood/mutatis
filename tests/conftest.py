"""Shared test setup.

Runs JAX on the CPU, and turns on runtime checking of the array shapes stated in mutatis's function signatures. The check
must be installed before any mutatis module is imported, which is why it happens here.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jaxtyping import install_import_hook  # noqa: E402

install_import_hook("mutatis", "beartype.beartype")

import jax  # noqa: E402

# Keep compiled functions on disk between test runs; compiling, not computing, dominates the time of most tests.
jax.config.update("jax_compilation_cache_dir", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".jax_cache"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.2)
