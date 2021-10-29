import jax
import jax.numpy as jnp
from .lbfgs import _minimize_lbfgs
from .algos import lbfgs_inverse_hessian_factors, lbfgs_sample, ELBO


def pathfinder(rng_key, logp_fn, x0, maxiter, maxcor, M, output='best'):
    """
    Pathfinder variational inference algorithm.
    It implements algorithm in figure 3 of:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782

    Parameters
    ----------
        rng_key : array
            Jax random RPNG key
        logp_fn : function
            log densify function of target distribution to take approximate samples from
        x0 : array
            initial point for the L-BFGS optimization routine
        maxiter : int
            maximum iteration of the L-BFGS optimization routine
        maxcor : int
            maximum number of metric correction of the the L-BFGS optimization routine
        M : int
            number of samples to draw
        output : str
            either 'best' to output only iteration that minimize ELBO, or 'all' to output all of them

    Returns
    -------
        if output='best' tuple with (best elbo, x value for iteration with best elbo, sample from iteration with best elbo)
        if outout='all' tuple containing elbo, x-values and samples for all interations

    """
    objective_fn = lambda x: - logp_fn(x).sum()
    status, history = _minimize_lbfgs(objective_fn, x0, maxiter, maxcor=maxcor)

    x,  g, alpha_scalar = history.x, history.g, history.gamma
    s, z = jnp.diff(x, axis=-2), jnp.diff(g, axis=-2)
    rng_keys = jax.random.split(rng_key, status.k)

    best_elbo, best_x, best_phis = -jnp.inf, None, None
    elbos, xs, phis = [], [], []
    for i in range(status.k+1):
        J = jnp.minimum(maxcor, i)
        S, Z = s[i-J:i].T, z[i-J:i].T
        elbo, phi = _pathfinder_inner_step(rng_keys[i], logp_fn,
                                           x[i], g[i],
                                           S, Z, alpha_scalar[i],
                                           M)
        if elbo > best_elbo and output == 'best':
            best_elbo, best_x, best_phis = elbo, x[i], phi
        else:
            elbos.append(elbo)
            xs.append(x[i])
            phis.append(phi)

    if output == 'best':
        return best_elbo, best_x, best_phis
    elif output == 'all':
        return elbos, xs, phis


def _pathfinder_inner_step(rng_key, logp_fn, theta, theta_grad,
                           S, Z, alpha_scalar, M):
    """
    Pathfinder algoritm internal loop.
    It implements point 4 of figure 3 algorithm of:

    Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782
    """

    alpha = alpha_scalar * jnp.ones(theta.shape[-1])
    beta, gamma = lbfgs_inverse_hessian_factors(S, Z, alpha)

    phi, logq, logdet = lbfgs_sample(rng_key, M, theta, theta_grad,
                                     alpha, beta, gamma)
    logp = logp_fn(phi)

    elbo = ELBO(logp, logq)
    return elbo, phi
