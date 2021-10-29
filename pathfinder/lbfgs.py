# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Limited-Memory Broyden-Fletcher-Goldfarb-Shanno minimization algorithm.
"""
## TAKEN FROM https://github.com/google/jax/blob/main/jax/_src/scipy/optimize/_lbfgs.py
## and added functionality to return optimization path

from collections import namedtuple
import jax
import jax.numpy as jnp
from jax import lax
from jax._src.scipy.optimize.line_search import line_search
from jax._src.scipy.optimize._lbfgs import LBFGSResults, _dot, _two_loop_recursion, _update_history_vectors, _update_history_scalars

LBFGSHistory = namedtuple("LBFGSHistory",
                          ["x", "f", "g", "gamma"])

def _minimize_lbfgs(
    fun,
    x0,
    maxiter=100,
    norm=jnp.inf,
    maxcor=10,
    ftol=2.220446049250313e-09,
    gtol=1e-05,
    maxfun=None,
    maxgrad=None,
    maxls=20,
):
    """
    Minimize a function using L-BFGS
  
    Implements the L-BFGS algorithm from
      Algorithm 7.5 from Wright and Nocedal, 'Numerical Optimization', 1999, pg. 176-185
    And generalizes to complex variables from
       Sorber, L., Barel, M.V. and Lathauwer, L.D., 2012.
       "Unconstrained optimization of real functions in complex variables"
       SIAM Journal on Optimization, 22(3), pp.879-898.
  
    Args:
      fun: function of the form f(x) where x is a flat ndarray and returns a real scalar.
        The function should be composed of operations with vjp defined.
      x0: initial guess
      maxiter: maximum number of iterations
      norm: order of norm for convergence check. Default inf.
      maxcor: maximum number of metric corrections ("history size")
      ftol: terminates the minimization when `(f_k - f_{k+1}) < ftol`
      gtol: terminates the minimization when `|g_k|_norm < gtol`
      maxfun: maximum number of function evaluations
      maxgrad: maximum number of gradient evaluations
      maxls: maximum number of line search steps (per iteration)
  
    Returns:
      Optimization results.
    """
    d = len(x0)
    dtype = jnp.dtype(x0)
  
    # ensure there is at least one termination condition
    if (maxiter is None) and (maxfun is None) and (maxgrad is None):
        maxiter = d * 200
  
    # set others to inf, such that >= is supported
    if maxfun is None:
        maxfun = jnp.inf
    if maxgrad is None:
        maxgrad = jnp.inf
  
    # initial evaluation
    f_0, g_0 = jax.value_and_grad(fun)(x0)
    state_initial = LBFGSResults(
      converged=False,
      failed=False,
      k=0,
      nfev=1,
      ngev=1,
      x_k=x0,
      f_k=f_0,
      g_k=g_0,
      s_history=jnp.zeros((maxcor, d), dtype=dtype),
      y_history=jnp.zeros((maxcor, d), dtype=dtype),
      rho_history=jnp.zeros((maxcor,), dtype=dtype),
      gamma=1.,
      status=0,
      ls_status=0,
    )

    history_initial = LBFGSHistory(
      x=_update_history_vectors(jnp.zeros((maxiter + 2, d), dtype=dtype),
                                x0),
      f=_update_history_scalars(jnp.zeros(maxiter + 2, dtype=dtype),
                                f_0),
      g=_update_history_vectors(jnp.zeros((maxiter + 2, d), dtype=dtype),
                                g_0),
      gamma=_update_history_scalars(jnp.zeros(maxiter + 2, dtype=dtype),
                                state_initial.gamma)
      )

    def cond_fun(args):
        state, history = args
        return (~(state.converged)) & (~(state.failed))
 
    def body_fun(args):
        state, history = args
        # find search direction
        p_k = _two_loop_recursion(state)
 
        # line search
        ls_results = line_search(
          f=fun,
          xk=state.x_k,
          pk=p_k,
          old_fval=state.f_k,
          gfk=state.g_k,
          maxiter=maxls,
        )
  
        # evaluate at next iterate
        s_k = ls_results.a_k * p_k
        x_kp1 = state.x_k + s_k
        f_kp1 = ls_results.f_k
        g_kp1 = ls_results.g_k
        y_k = g_kp1 - state.g_k
        rho_k_inv = jnp.real(_dot(y_k, s_k))
        rho_k = jnp.reciprocal(rho_k_inv)
        gamma = rho_k_inv / jnp.real(_dot(jnp.conj(y_k), y_k))
  
        # replacements for next iteration
        status = 0
        status = jnp.where(state.f_k - f_kp1 < ftol, 4, status)
        status = jnp.where(state.ngev >= maxgrad, 3, status)  # type: ignore
        status = jnp.where(state.nfev >= maxfun, 2, status)  # type: ignore
        status = jnp.where(state.k >= maxiter, 1, status)  # type: ignore
        status = jnp.where(ls_results.failed, 5, status)

        converged = jnp.linalg.norm(g_kp1, ord=norm) < gtol

        state = state._replace(
          converged=converged,
          failed=(status > 0) & (~converged),
          k=state.k + 1,
          nfev=state.nfev + ls_results.nfev,
          ngev=state.ngev + ls_results.ngev,
          x_k=x_kp1,
          f_k=f_kp1,
          g_k=g_kp1,
          s_history=_update_history_vectors(history=state.s_history, new=s_k),
          y_history=_update_history_vectors(history=state.y_history, new=y_k),
          rho_history=_update_history_scalars(history=state.rho_history, new=rho_k),
          gamma=gamma,
          status=jnp.where(converged, 0, status),
          ls_status=ls_results.status,
        )

        history = history._replace(
          x=_update_history_vectors(history=history.x, new=x_kp1),
          f=_update_history_scalars(history=history.f, new=f_kp1),
          g=_update_history_vectors(history=history.g, new=g_kp1),
          gamma=_update_history_scalars(history=history.gamma, new=gamma)
          )

        return (state, history)

    state, history = lax.while_loop(cond_fun, body_fun, (state_initial, history_initial))
    history = history._replace(
             x=history.x[-state.k-1:],
             f=history.f[-state.k-1:],
             g=history.g[-state.k-1:],
             gamma=history.gamma[-state.k-1:]
             )

    return state, history

