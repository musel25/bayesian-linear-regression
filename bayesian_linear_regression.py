"""
Bayesian (Gaussian) Linear Regression — educational implementation

This module implements, documents, and demos three steps typically requested in
an introductory Gaussian Linear Regression (GLR) exercise:

1) Generate synthetic datasets from a user-defined model function.
2) Implement GLR prediction given training data, noise covariance, and prior.
3) Explore several noise/prior settings and plot credibility intervals.

Notes
-----
- The code prioritizes clarity over micro-optimizations; it deliberately
  uses `np.linalg.inv` to emphasize the underlying algebra. For production, one
  would prefer numerically stable solves (e.g., via Cholesky factorization) and
  avoid explicit matrix inversion when possible.
- Shapes are kept explicit and small helper utilities (_to_2d_column,
  _augment_bias, _as_cov_matrix) make intent clear and reduce boilerplate.

"""

import numpy as np
import scipy as sp
import scipy.stats as spst
import matplotlib.pyplot as plt


def _to_2d_column(x):
    """Return ``x`` as a 2D column array of shape (n, 1).

    This utility keeps the rest of the implementation simple and explicit about
    shapes. It accepts scalars, 1D arrays, or already 2D inputs and guarantees a
    column output when possible.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _augment_bias(X):
    """Prepend a column of ones to ``X`` to model an intercept term.

    In linear regression, the intercept can be represented by adding a constant
    (bias) feature equal to 1 for every sample. Given ``X`` of shape (n, d), the
    returned matrix has shape (n, d+1) with the first column equal to 1.
    """
    X = _to_2d_column(X) if X.ndim == 1 or X.shape[1] == 1 else np.asarray(X)
    ones = np.ones((X.shape[0], 1))
    return np.concatenate([ones, X], axis=1)


def make_synthetic_data(model_fn, n_train=25, n_test=200, x_range=(-1.0, 1.0), noise_std=0.2, seed=0):
    """Step 1 — Generate synthetic training/test data from a user model.

    Parameters
    ----------
    model_fn : callable
        A user-defined function implementing the ground-truth relation.
        It receives an array of shape (n, 1) and returns an array of shape
        (n, 1). Examples: linear ``lambda x: a + b*x`` or nonlinear functions.
    n_train : int
        Number of training samples to draw uniformly in ``x_range``.
    n_test : int
        Number of test points (regular grid) to evaluate the noiseless model.
    x_range : tuple(float, float)
        Domain bounds for input sampling, inclusive.
    noise_std : float
        Standard deviation of additive i.i.d. Gaussian noise applied to
        training outputs. The test outputs are left noiseless for reference.
    seed : int
        Seed for the RNG to make the dataset reproducible.

    Returns
    -------
    Xtr : (n_train, 1) ndarray
        Training inputs sampled uniformly from ``x_range``.
    ytr : (n_train, 1) ndarray
        Noisy training outputs: ``model_fn(Xtr) + Normal(0, noise_std^2)``.
    Xte : (n_test, 1) ndarray
        Test inputs on a regular grid over ``x_range``.
    yte : (n_test, 1) ndarray
        Noise-free model values at ``Xte`` for reference/plotting.

    Implementation details
    ----------------------
    - Inputs are shaped as explicit column vectors.
    - Training x-values are drawn from a uniform distribution; test x-values are
      a dense linspace for smooth plotting/comparison.
    - Only training outputs receive noise, matching common regression setups.
    """
    rng = np.random.default_rng(seed)
    Xtr = rng.uniform(x_range[0], x_range[1], size=(n_train, 1))
    Xte = np.linspace(x_range[0], x_range[1], n_test).reshape(-1, 1)
    ytr = model_fn(Xtr) + rng.normal(0.0, noise_std, size=(n_train, 1))
    yte = model_fn(Xte)
    return Xtr, ytr, Xte, yte


def _as_cov_matrix(value, size):
    """Return a covariance matrix of shape (size, size) from various inputs.

    Accepts:
    - scalar ``value``: returns ``value * I`` (isotropic covariance),
    - 1D array of length ``size``: returns ``diag(value)``,
    - 2D array with shape (size, size): returned as-is after validation.

    Raises ``ValueError`` if shapes are incompatible.
    """
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.eye(size) * float(arr)
    if arr.ndim == 1:
        if arr.shape[0] != size:
            raise ValueError("Vector length does not match required size for covariance.")
        return np.diag(arr)
    if arr.shape[0] != size or arr.shape[1] != size:
        raise ValueError("Covariance matrix has incompatible shape.")
    return arr


def predGLR(Xpr, Xtr, ytr, Sign, Sigp, alpha=0.95, include_noise=True):
    """Step 2 — Gaussian Linear Regression predictions with credibility intervals.

    Given training design matrix ``Xtr`` and targets ``ytr``, a noise covariance
    ``Σ_n`` and a Gaussian prior on weights ``w ~ N(0, Σ_p)``, this function
    computes the posterior over weights and the predictive distribution at test
    inputs ``Xpr``. It returns point predictions, predictive covariance, and
    symmetric credibility intervals at level ``alpha``.

    Model and posterior
    -------------------
    - Likelihood: ``y | X, w ~ N(X w, Σ_n)``
    - Prior:      ``w ~ N(0, Σ_p)``
    - Posterior:  ``p(w | X, y) = N(m_N, S_N)`` with
        ``S_N = (Σ_p^{-1} + X^T Σ_n^{-1} X)^{-1}``
        ``m_N = S_N X^T Σ_n^{-1} y``

    Predictive distribution at new inputs ``Xpr`` (with intercept term added):
    - Latent function:   ``f_* = Xpr m_N``, cov ``Cov[f_*] = Xpr S_N Xpr^T``
    - With observation noise (optional): ``y_* = f_* + ε``, with
      ``Cov[y_*] = Cov[f_*] + Σ_{n,*}`` where ``Σ_{n,*}`` is typically
      a diagonal matrix with the same (average) variance as training noise.

    Parameters
    ----------
    Xpr : (n_pr, d) or (n_pr, 1) ndarray
        Test inputs. A bias column is automatically added internally.
    Xtr : (n_tr, d) or (n_tr, 1) ndarray
        Training inputs. A bias column is automatically added internally.
    ytr : (n_tr,) or (n_tr, 1) ndarray
        Training targets. Converted to column vector.
    Sign : float | (n_tr,) | (n_tr, n_tr)
        Noise covariance for training data ``Σ_n``. A scalar is interpreted as
        isotropic noise; a vector as per-sample variances; a full matrix is
        used as given. For predictive noise, we add ``σ^2 I`` with
        ``σ^2`` equal to ``Sign`` if scalar, the average of the vector if 1D,
        or the average diagonal element if a full matrix.
    Sigp : float | (d+1,) | (d+1, d+1)
        Prior covariance for weights, including the intercept term. If scalar,
        uses ``Sigp * I``; if vector, uses ``diag(Sigp)``; if full matrix, it is
        used directly. The shape must match the augmented feature dimension
        (``d+1`` due to the bias column).
    alpha : float, default 0.95
        Credibility mass for symmetric intervals (e.g., 0.95 → 95%).
    include_noise : bool, default True
        If True, intervals cover observations (includes noise). If False,
        intervals cover the latent function (excludes noise).

    Returns
    -------
    ypr_mean : (n_pr, 1) ndarray
        Predictive mean at ``Xpr``.
    ypr_cov : (n_pr, n_pr) ndarray
        Predictive covariance matrix.
    ypr_inf : (n_pr,) ndarray
        Lower bound of the symmetric credibility interval at level ``alpha``.
    ypr_sup : (n_pr,) ndarray
        Upper bound of the symmetric credibility interval at level ``alpha``.

    Implementation details
    ----------------------
    - We explicitly augment inputs with a bias column via ``_augment_bias`` so
      that the prior and posterior cover the intercept.
    - We convert ``Sign`` and ``Sigp`` to full covariance matrices using
      ``_as_cov_matrix`` for consistent linear algebra.
    - For pedagogical clarity we use ``np.linalg.inv`` at a few places to show
      the algebra; more stable alternatives would solve linear systems without
      forming inverses explicitly.
    - Symmetric credibility intervals are computed as
      ``mean ± z_{alpha/2} * sqrt(diag(cov))`` with ``z`` the Normal quantile.
    """
    Xtr = _augment_bias(np.asarray(Xtr))
    Xpr = _augment_bias(np.asarray(Xpr))
    ytr = _to_2d_column(ytr)

    n_tr, d = Xtr.shape
    n_pr = Xpr.shape[0]

    Sig_n = _as_cov_matrix(Sign, n_tr)
    Sig_p = _as_cov_matrix(Sigp, d)

    Sig_n_inv = np.linalg.inv(Sig_n)
    Sig_p_inv = np.linalg.inv(Sig_p)

    # Posterior over weights: S_N = (Σp^{-1} + X^T Σn^{-1} X)^{-1}
    # and m_N = S_N X^T Σn^{-1} y
    A = Sig_p_inv + Xtr.T @ Sig_n_inv @ Xtr
    S_N = np.linalg.inv(A)
    m_N = S_N @ Xtr.T @ Sig_n_inv @ ytr

    # Latent predictive mean and covariance (function values without noise)
    ypr_mean = Xpr @ m_N
    ypr_cov_latent = Xpr @ S_N @ Xpr.T

    if include_noise:
        # Approximate observation noise on predictions by adding σ^2 I, where
        # σ^2 matches the (average) training noise variance.
        if np.ndim(Sign) == 0:
            Sig_n_pr = float(Sign) * np.eye(n_pr)
        elif np.ndim(Sign) == 1:
            sigma2 = float(np.mean(np.asarray(Sign)))
            Sig_n_pr = sigma2 * np.eye(n_pr)
        else:
            sigma2 = float(np.mean(np.diag(_as_cov_matrix(Sign, n_tr))))
            Sig_n_pr = sigma2 * np.eye(n_pr)
        ypr_cov = ypr_cov_latent + Sig_n_pr
    else:
        ypr_cov = ypr_cov_latent

    var = np.clip(np.diag(ypr_cov), a_min=0.0, a_max=None)
    std = np.sqrt(var)
    z = spst.norm.ppf(0.5 + alpha / 2.0)
    ypr_inf = ypr_mean.reshape(-1) - z * std
    ypr_sup = ypr_mean.reshape(-1) + z * std

    return [ypr_mean, ypr_cov, ypr_inf, ypr_sup]


def _demo():
    """Step 3 — Demo: vary noise/prior and plot credibility intervals.

    This function:
    - defines a ground-truth linear model, ``true_model(x) = 1 + 2x``,
    - generates training and dense plotting sets via ``make_synthetic_data``,
    - runs ``predGLR`` for several observation noise levels and prior scales,
    - plots the training data, predictive mean, credibility intervals, and the
      true (noise-free) function for reference.

    The plotting snippet mirrors common coursework instructions, e.g.:

        fig, ax = plt.subplots()
        ax.plot(Xtr, ytr, 'k+', label='train data')
        ax.plot(Xpl, ypl_ave, label='estimates')
        ax.fill_between(Xpl, ypl_inf.reshape(-1), ypl_sup.reshape(-1),
                        color='lightblue', label='95% credibility interval')
        ax.legend()

    """
    def true_model(x):
        x = _to_2d_column(x)
        return 1.0 + 2.0 * x

    noise_levels = [0.05, 0.2, 0.5]
    prior_scales = [0.1, 1.0, 10.0]
    alpha = 0.95

    fig, axes = plt.subplots(
        len(noise_levels), len(prior_scales),
        figsize=(4 * len(prior_scales), 3 * len(noise_levels)),
        sharex=True, sharey=True
    )

    for i, sigma in enumerate(noise_levels):
        Xtr, ytr, Xpl, ypl_true = make_synthetic_data(true_model, n_train=20, n_test=200, noise_std=sigma, seed=42 + i)
        for j, ps in enumerate(prior_scales):
            Sigp = np.eye(2) * ps
            ypl_ave, ypl_cov, ypl_inf, ypl_sup = predGLR(Xpl, Xtr, ytr, Sign=sigma**2, Sigp=Sigp, alpha=alpha, include_noise=True)
            ax = axes[i, j] if axes.ndim == 2 else axes[max(i, j)]
            ax.plot(Xtr.reshape(-1), ytr.reshape(-1), 'k+', label='train data')
            ax.plot(Xpl.reshape(-1), ypl_ave.reshape(-1), label='estimates')
            ax.fill_between(Xpl.reshape(-1), ypl_inf.reshape(-1), ypl_sup.reshape(-1), color='lightblue', label=str(int(alpha * 100)) + '% credibility interval')
            ax.plot(Xpl.reshape(-1), ypl_true.reshape(-1), 'r--', linewidth=1.0, label='true model')
            ax.set_title(f'noise={sigma}, prior_scale={ps}')
            if i == len(noise_levels) - 1:
                ax.set_xlabel('x')
            if j == 0:
                ax.set_ylabel('y')
            if i == 0 and j == 0:
                ax.legend(loc='best', fontsize=8)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    _demo()
