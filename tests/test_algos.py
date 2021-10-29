import numpy as np
import jax.random
import jax.numpy as jnp
from jax._src.scipy.optimize._lbfgs import _two_loop_recursion
from pathfinder.lbfgs import _minimize_lbfgs
from pathfinder.algos import (lbfgs_inverse_hessian_factors,
                              lbfgs_inverse_hessian_formula_1,
                              lbfgs_inverse_hessian_formula_2,
                              lbfgs_sample)


def create_test_data(n, d):
    betas = np.random.randn(d)
    X = np.random.randn(n, d)
    y = X@betas + np.random.rand(n)
    fun = lambda b: jnp.linalg.norm(X@b-y).sum()
    b0 = jnp.array(np.random.randn(d))
    return fun, b0


def test_inverse_hessian():

    for i in range(3):
        maxcor, d, n = 10, 10, 1000

        fun, b0 = create_test_data(n, d)
        status, history = _minimize_lbfgs(fun, b0, maxiter=i, maxcor=maxcor)
        pk = _two_loop_recursion(status)

        J = min(status.k, maxcor)

        s, z = jnp.diff(history.x.T), jnp.diff(history.g.T)
        S = s[:, -J:]
        Z = z[:, -J:]

        alpha = history.gamma[status.k-1+J]*np.ones(d)
        beta, gamma = lbfgs_inverse_hessian_factors(S, Z, alpha)
        inv_hess_1 = lbfgs_inverse_hessian_formula_1(alpha, beta, gamma)
        inv_hess_2 = lbfgs_inverse_hessian_formula_2(alpha, beta, gamma)

        assert jnp.allclose(inv_hess_1, inv_hess_2)
        assert jnp.allclose(pk, -inv_hess_1 @ history.g[-1, :], 1e-4, 1e-4)
        assert jnp.allclose(pk, -inv_hess_2 @ history.g[-1, :], 1e-4, 1e-4)


def test_sample():

    rng_key = jax.random.PRNGKey(10)
    maxcor, d, n = 10, 10, 1000

    fun, b0 = create_test_data(n, d)
    status, history = _minimize_lbfgs(fun, b0, maxiter=100, maxcor=maxcor)

    J = min(maxcor, status.k)
    s, z = jnp.diff(history.x.T), jnp.diff(history.g.T)
    S = s[:, -J:]
    Z = z[:, -J:]
    M = 200_000

    alpha = history.gamma[-1] * jnp.ones(d)
    beta, gamma = lbfgs_inverse_hessian_factors(S, Z, alpha)
    phi, logq, logdet = lbfgs_sample(rng_key, M, history.x[-1], history.g[-1],
                                     alpha, beta, gamma)

    mu = history.x[-1] + jnp.diag(alpha) @ history.g[-1] +\
        beta @ gamma @ beta.T @ history.g[-1]
    inv_hess = lbfgs_inverse_hessian_formula_1(alpha, beta, gamma)

    inv_hess_hat = jnp.cov(phi, rowvar=False)
    logq_1 = jax.scipy.stats.multivariate_normal.logpdf(phi, mu, inv_hess)

    assert jnp.linalg.norm(mu-phi.mean(0)) < 10.
    assert jnp.allclose(jnp.linalg.det(inv_hess_hat-inv_hess), 0.)
    assert jnp.allclose(logdet, jnp.log(jnp.linalg.det(inv_hess)), rtol=1e-3)
    assert jnp.allclose(logq, logq_1, rtol=1e-4, atol=1e-4)
