"""JAX settings shared by every layer that uses JAX. Imported first by each such layer."""

import jax

# Solving these models means deciding which side of a stability boundary an eigenvalue falls on; 32-bit floats can get that wrong.
jax.config.update("jax_enable_x64", True)
