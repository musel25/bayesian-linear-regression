# Bayesian Linear Regression (GLR)

This repository contains a clear, well-documented implementation of Bayesian (Gaussian) Linear Regression and a small demo to visualize predictive credibility intervals while varying observation noise and prior covariance.

Note: this is an exercise from class AOS1 by Benjamin Quost.

## What’s Implemented

- Step 1: Synthetic data generation from a user-defined model function.
- Step 2: GLR predictions with a Gaussian prior over weights and Gaussian noise, returning predictive mean, covariance, and credibility intervals.
- Step 3: A demo that sweeps several noise levels and prior covariances, plotting training data, predictive mean, and credibility intervals.

See `bayesian_linear_regression.py` for detailed docstrings explaining how each step is carried out and the linear algebra behind it.

## Quick Start

- Run the demo to reproduce the plots for multiple noise/prior settings:

  ```bash
  python bayesian_linear_regression.py
  ```

## Step 1 — Generate Synthetic Data

Function: `make_synthetic_data(model_fn, n_train, n_test, x_range, noise_std, seed)`

- Draws training inputs uniformly from `x_range` and adds i.i.d. Gaussian noise with standard deviation `noise_std` to training outputs.
- Creates a regular grid of test inputs and evaluates the noise-free `model_fn` on them for reference/plotting.
- Shapes are explicit (column vectors), and an intercept term is handled later by augmenting inputs with a bias column.

## Step 2 — GLR Predictions + Credibility Intervals

Function: `predGLR(Xpr, Xtr, ytr, Sign, Sigp, alpha=0.95, include_noise=True)`

Model assumptions:
- Likelihood: $y \mid X, w \sim \mathcal{N}(Xw, \Sigma_n)$
- Prior: $w \sim \mathcal{N}(0, \Sigma_p)$

Posterior over weights:
$$
S_N = \bigl(\Sigma_p^{-1} + X^\top \Sigma_n^{-1} X\bigr)^{-1}, \quad
m_N = S_N X^\top \Sigma_n^{-1} y.
$$

Predictive distribution at new inputs $X_{*}$ (bias column auto-added):
- Latent mean/cov: $f_{*} = X_{*} m_N$, $\operatorname{Cov}[f_{*}] = X_{*} S_N X_{*}^\top$
- With observation noise (optional): $\operatorname{Cov}[y_{*}] = \operatorname{Cov}[f_{*}] + \Sigma_{n,*}$,
  where typically $\Sigma_{n,*} = \sigma^2 I$ and $\sigma^2$ matches the (average) training noise variance.

Credibility intervals (symmetric) at level $\alpha$ use the normal quantile $z$:
$$
\text{mean} \;\pm\; z\, \sqrt{\operatorname{diag}(\text{cov})}, \quad
z = \Phi^{-1}\!\bigl(0.5 + \tfrac{\alpha}{2}\bigr).
$$

Implementation notes:
- Uses `np.linalg.inv` for pedagogical clarity (algebra is explicit). In production, prefer solving linear systems or Cholesky for numerical stability.
- Accepts scalar, diagonal (1D), or full covariance matrices for both `Σ_n` and `Σ_p` (see `_as_cov_matrix`).

## Math Derivation

We assume a linear model with intercept handled by augmenting a 1-column of ones:
$$
y \mid X, w \sim \mathcal{N}(Xw, \Sigma_n),\qquad w \sim \mathcal{N}(0, \Sigma_p).
$$

Log posterior (up to constants):
$$
\log p(w\mid X,y) = -\tfrac{1}{2} (y - Xw)^\top \Sigma_n^{-1} (y - Xw)
                     -\tfrac{1}{2} w^\top \Sigma_p^{-1} w + C.
$$

Completing the square in $w$ gives
$$
S_N^{-1} = \Sigma_p^{-1} + X^\top \Sigma_n^{-1} X,\qquad
S_N^{-1} m_N = X^\top \Sigma_n^{-1} y \;\Rightarrow\; m_N = S_N X^\top \Sigma_n^{-1} y,
$$
so $p(w\mid X,y) = \mathcal{N}(m_N, S_N)$ with
$$
S_N = \bigl(\Sigma_p^{-1} + X^\top \Sigma_n^{-1} X\bigr)^{-1}.
$$

Predictive distribution at $X_{*}$:
$$
f_{*} \mid X_{*}, X, y \sim \mathcal{N}(X_{*} m_N,\; X_{*} S_N X_{*}^\top),
$$
and with observation noise
$$
y_{*} \mid X_{*}, X, y \sim \mathcal{N}\bigl(X_{*} m_N,\; X_{*} S_N X_{*}^\top + \Sigma_{n,*}\bigr).
$$

Credibility intervals (symmetric, level $\alpha$):
$$
\text{mean} \pm z_{\alpha/2}\, \sqrt{\operatorname{diag}(\text{cov})},
\quad z_{\alpha/2} = \Phi^{-1}\!\bigl(0.5 + \tfrac{\alpha}{2}\bigr).
$$

## Step 3 — Plotting Example

Minimal plotting snippet (matches common coursework instructions):

```python
fig, ax = plt.subplots()
ax.plot(Xtr, ytr, 'k+', label='train data')
ax.plot(Xpl, ypl_ave, label='estimates')
ax.fill_between(Xpl, ypl_inf.reshape(-1), ypl_sup.reshape(-1),
                color='lightblue', label=str(int(alpha*100))+'% credibility interval')
ax.legend()
```

The built-in demo (`python bayesian_linear_regression.py`) generates synthetic data for several noise levels and sweeps over different prior scales, then plots training points, the predictive mean, the credibility band, and the true model.

## Files

- `bayesian_linear_regression.py`: Implementation with extensive docstrings and a demo.
