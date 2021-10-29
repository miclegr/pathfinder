import jax.random
import jax.numpy as jnp
from functools import partial


def lbfgs_inverse_hessian_factors(S, Z, alpha):
    """
    Calculates factors for inverse hessian factored representation.
    It implements algorithm of figure 7 in:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782
    """
    J = S.shape[1]
    StZ = S.T @ Z
    R = jnp.triu(StZ)

    eta = jnp.diag(StZ)

    beta = jnp.hstack([jnp.diag(alpha) @ Z, S])

    minvR = -jnp.linalg.inv(R)
    alphaZ = jnp.diag(jnp.sqrt(alpha)) @ Z
    block_dd = minvR.T @ (alphaZ.T @ alphaZ + jnp.diag(eta)) @ minvR
    gamma = jnp.block([[jnp.zeros((J, J)), minvR],
                       [minvR.T, block_dd]])
    return beta, gamma


def lbfgs_inverse_hessian_formula_1(alpha, beta, gamma):
    """
    Calculates inverse hessian from factors as in figure 7:

    """
    return jnp.diag(alpha) + beta @ gamma @ beta.T


def lbfgs_inverse_hessian_formula_2(alpha, beta, gamma):
    """
    Calculates inverse hessian from factors as in formula II.1 of:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782
    """
    N = alpha.shape[0]
    dsqrt_alpha = jnp.diag(jnp.sqrt(alpha))
    idsqrt_alpha = jnp.diag(1/jnp.sqrt(alpha))
    return dsqrt_alpha @ (jnp.eye(N) +
                          idsqrt_alpha @ beta @ gamma @ beta.T @ idsqrt_alpha
                          ) @ dsqrt_alpha


def lbfgs_sample(rng_key, M, theta, grad_theta, alpha, beta, gamma):
    """
    Draws approximate samples of target distribution.
    It implements algorith of figure 8 in:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782
    """

    Q, R = jnp.linalg.qr(jnp.diag(jnp.sqrt(1/alpha)) @ beta)
    N, J = beta.shape[0], R.shape[0]
    L = jnp.linalg.cholesky(jnp.eye(J) + R @ gamma @ R.T)

    logdet = jnp.log(jnp.prod(alpha)) + 2*jnp.log(jnp.linalg.det(L))

    mu = theta + jnp.diag(alpha) @ grad_theta + \
        beta @ gamma @ beta.T @ grad_theta

    u = jax.random.normal(rng_key, (M, N, 1))

    phi = (mu[None, :, None] + jnp.diag(jnp.sqrt(alpha))@(
        (Q @ L @ Q.T) @ u + u - (Q @ Q.T @ u)
        ))[:, :, 0]

    logq = -.5 * (logdet + (u**2).sum([-1, -2]) + N*jnp.log(2.*jnp.pi))
    return phi, logq, logdet


def ELBO(logp, logq):
    """
    Calculates approximate ELBO using monte carlo.
    It implements algorith of figure 9 in:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782
    """
    return (logp - logq).mean()
