import jax.random
import jax.numpy as jnp
import numpy as np


def lbfgs_inverse_hessian_factors(S, Z, alpha):
    J = S.shape[1]
    StZ = S.T @ Z
    R = np.triu(StZ)

    eta = jnp.diag(StZ)

    beta = jnp.hstack([jnp.diag(alpha) @ Z, S])

    minvR = -jnp.linalg.inv(R)
    alphaZ = jnp.diag(jnp.sqrt(alpha)) @ Z
    block_dd = minvR.T @ (alphaZ.T @ alphaZ + jnp.diag(eta)) @ minvR
    gamma = jnp.block([[jnp.zeros((J, J)), minvR],
                       [minvR.T, block_dd]])
    return beta, gamma


def lbfgs_inverse_hessian_formula_1(alpha, beta, gamma):
    return jnp.diag(alpha) + beta @ gamma @ beta.T


def lbfgs_inverse_hessian_formula_2(alpha, beta, gamma):
    N = alpha.shape[0]
    dsqrt_alpha = jnp.diag(jnp.sqrt(alpha))
    idsqrt_alpha = jnp.diag(1/jnp.sqrt(alpha))
    return dsqrt_alpha @ (jnp.eye(N) +
                          idsqrt_alpha @ beta @ gamma @ beta.T @ idsqrt_alpha
                          ) @ dsqrt_alpha


def lbfgs_sample(rng_key, M, theta, grad_theta, alpha, beta, gamma):

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
    return phi, logdet
