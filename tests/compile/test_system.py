"""Compiling specifications into equation systems."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mutatis.compile.expand import CompileError
from mutatis.compile.system import compile_model
from mutatis.spec import edits as e
from mutatis.spec.model import Block, Parameter
from mutatis.spec.yaml_io import load_model
from tests.spec.fixtures import toy


def jacobians(system, params):
    n, k, m = system.n_variables, len(system.shock_labels), len(system.outside_labels)
    zeros = [jnp.zeros(n), jnp.zeros(n), jnp.zeros(n), jnp.zeros(k), jnp.zeros(m)]
    return jax.jacfwd(system.residual, argnums=(0, 1, 2, 3, 4))(*zeros, params), system.residual(*zeros, params)


def test_forward_looking_toy_matrices():
    system = compile_model(toy("forward_looking"))
    params = {"demand.rho": jnp.array(0.8), "demand.sigma_x": jnp.array(1.0), "price.a": jnp.array(0.5)}
    (A, B, C, D, G), f = jacobians(system, params)
    np.testing.assert_allclose(A, [[0, 0], [0, -0.5]])
    np.testing.assert_allclose(B, [[1, 0], [-1, 1]])
    np.testing.assert_allclose(C, [[-0.8, 0], [0, 0]])
    np.testing.assert_allclose(D, [[-1], [0]])
    assert G.shape == (2, 0) and float(jnp.abs(f).max()) == 0.0


def test_two_variable_matrices():
    system = compile_model(toy("two_variable"))
    c, Bm = jnp.array([0.1, -0.2]), jnp.array([[0.5, 0.1], [0.0, 0.3]])
    params = {"dynamics.c": c, "dynamics.B": Bm, "dynamics.Sigma": jnp.eye(2)}
    (A, B, C, D, G), f = jacobians(system, params)
    np.testing.assert_allclose(A, 0)
    np.testing.assert_allclose(B, np.eye(2))
    np.testing.assert_allclose(C, -Bm)
    np.testing.assert_allclose(D, -np.eye(2))
    np.testing.assert_allclose(f, -c)


def test_outside_variable_with_a_lag_becomes_a_state_copy():
    system = compile_model(toy("error_correction"))
    assert system.variable_labels == ("spend", "income@now")
    assert system.outside_labels == ("income",)
    assert system.outside["income"].series == {"income": "income"}
    params = {f"spending.{k}": jnp.array(v) for k, v in dict(alpha0=0.1, gamma=-0.2, beta1=0.9, alpha1=0.3, sigma=1.0).items()}
    lag = jnp.array([1.0, 2.0])   # spend, income one period ago
    now = jnp.array([1.5, 2.5])
    shock, outside = jnp.array([0.05]), jnp.array([2.5])
    residual = system.residual(jnp.zeros(2), now, lag, shock, outside, params)
    expected_spend = (1.5 - 1.0) - (0.1 - 0.2 * (1.0 - 0.9 * 2.0) + 0.3 * (2.5 - 2.0) + 0.05)
    np.testing.assert_allclose(residual, [expected_spend, 2.5 - 2.5], atol=1e-14)


def test_long_lags_get_a_chain_of_extra_variables():
    model = load_model("""
model: lags
blocks:
  b:
    outputs: [x]
    parameters: {r: {fixed: 0.5}}
    shocks: {e: "normal(0, 1)"}
    equations: ["x = r * x(-3) + e"]
data:
  x: {observes: "x - x(-2)", error: "normal(0, 1)"}
""")
    system = compile_model(model)
    assert system.variable_labels == ("x", "x@lag1", "x@lag2")
    assert system.presample == 3
    # The observation x - x(-2) uses the state now: x - x@lag2.
    np.testing.assert_allclose(system.observe(jnp.array([5.0, 4.0, 3.0]), {"b.r": 0.5}), [2.0])


def test_templates_expand_and_sums_follow_the_set():
    model = toy("sectors")
    system = compile_model(model)
    assert [lab for lab in system.variable_labels if lab.startswith("k[")] == ["k[n]", "k[m]"]
    bigger = compile_model(e.ExtendSet("sectors", ("h",)).apply(model))
    assert "k[h]" in bigger.variable_labels and "investment.u[h]" in bigger.shock_labels
    params = {k: jnp.zeros(p.shape) for k, p in bigger.parameters.items()}
    now = jnp.zeros(bigger.n_variables)
    for i, label in enumerate(bigger.variable_labels):
        if label.startswith("k["):
            now = now.at[i].set(1.0)
    residual = bigger.residual(now, now, now, jnp.zeros(4), jnp.zeros(0), params)
    total_row = bigger.variable_labels.index("K")  # one equation per output, in order, so K's equation is K's row
    np.testing.assert_allclose(residual[total_row], 0.0 - 3.0)


def test_untied_parameter_gets_its_own_key():
    model = e.Untie("investment.phi", "m", prior="normal(2, 0.5)").apply(toy("sectors"))
    system = compile_model(model)
    assert "investment.phi[m]" in system.parameters and "investment.phi" in system.parameters


def test_naive_expectations_remove_the_forward_looking_term():
    model = e.SetExpectations("naive").apply(toy("forward_looking"))
    system = compile_model(model)
    assert not system.has_expectations
    params = {"demand.rho": jnp.array(0.8), "demand.sigma_x": jnp.array(1.0), "price.a": jnp.array(0.5)}
    (A, B, C, D, G), f = jacobians(system, params)
    np.testing.assert_allclose(A, 0)
    np.testing.assert_allclose(C, [[-0.8, 0], [0, -0.5]])  # E p(+1) became p(-1)


def test_discounted_expectations_scale_the_term():
    model = load_model(toy_text("forward_looking").replace("expectations: rational", "expectations: discounted(m)\nparameters:\n  m: {fixed: 0.9}"))
    system = compile_model(model)
    params = {"demand.rho": jnp.array(0.8), "demand.sigma_x": jnp.array(1.0), "price.a": jnp.array(0.5), "m": jnp.array(0.9)}
    (A, *_), _ = jacobians(system, params)
    np.testing.assert_allclose(A, [[0, 0], [0, -0.45]])


def toy_text(name):
    from tests.spec.fixtures import MODELS
    return (MODELS / f"{name}.yaml").read_text()


def test_counting_is_checked_block_by_block():
    model = toy("forward_looking")
    price = model.blocks["price"]
    broken = e.ReplaceBlock("price", Block(inputs=price.inputs, outputs=price.outputs, parameters=price.parameters,
                                           equations=(price.equations[0], "p = x"))).apply(model)
    with pytest.raises(CompileError, match="price: 2 equation rows for 1 outputs"):
        compile_model(broken)


def test_future_values_must_be_marked_as_expectations():
    model = e.ReplaceEquation("price", 0, "p = x + a * p(+1)").apply(toy("forward_looking"))
    with pytest.raises(CompileError, match="must be inside E"):
        compile_model(model)


def test_nonlinear_equations_are_detected():
    model = e.ReplaceEquation("demand", 0, "x = rho * x(-1)^2 + e").apply(toy("forward_looking"))
    assert not compile_model(model).linear
    assert compile_model(toy("forward_looking")).linear
