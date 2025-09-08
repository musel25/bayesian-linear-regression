import numpy as np
import scipy as sp
import scipy.stats as spst
import matplotlib.pyplot as plt


def _to_2d_column(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _augment_bias(X):
    X = _to_2d_column(X) if X.ndim == 1 or X.shape[1] == 1 else np.asarray(X)
    ones = np.ones((X.shape[0], 1))
    return np.concatenate([ones, X], axis=1)


def make_synthetic_data(model_fn, n_train=25, n_test=200, x_range=(-1.0, 1.0), noise_std=0.2, seed=0):
    rng = np.random.default_rng(seed)
    Xtr = rng.uniform(x_range[0], x_range[1], size=(n_train, 1))
    Xte = np.linspace(x_range[0], x_range[1], n_test).reshape(-1, 1)
    ytr = model_fn(Xtr) + rng.normal(0.0, noise_std, size=(n_train, 1))
    yte = model_fn(Xte)
    return Xtr, ytr, Xte, yte


def _as_cov_matrix(value, size):
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
    Xtr = _augment_bias(np.asarray(Xtr))
    Xpr = _augment_bias(np.asarray(Xpr))
    ytr = _to_2d_column(ytr)

    n_tr, d = Xtr.shape
    n_pr = Xpr.shape[0]

    Sig_n = _as_cov_matrix(Sign, n_tr)
    Sig_p = _as_cov_matrix(Sigp, d)

    Sig_n_inv = np.linalg.inv(Sig_n)
    Sig_p_inv = np.linalg.inv(Sig_p)

    A = Sig_p_inv + Xtr.T @ Sig_n_inv @ Xtr
    S_N = np.linalg.inv(A)
    m_N = S_N @ Xtr.T @ Sig_n_inv @ ytr

    ypr_mean = Xpr @ m_N
    ypr_cov_latent = Xpr @ S_N @ Xpr.T

    if include_noise:
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
    def true_model(x):
        x = _to_2d_column(x)
        return 1.0 + 2.0 * x

    noise_levels = [0.05, 0.2, 0.5]
    prior_scales = [0.1, 1.0, 10.0]
    alpha = 0.95

    fig, axes = plt.subplots(len(noise_levels), len(prior_scales), figsize=(4 * len(prior_scales), 3 * len(noise_levels)), sharex=True, sharey=True)

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
