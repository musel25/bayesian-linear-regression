import numpy as np
import scipy as sp
import scipy.stats as spst
import matplotlib.pyplot as plt

def true_model(x):
    x = _to_2d_column(x)
    return 1.0 + 2.0 * x

def _to_2d_column(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x

model_fn = true_model
n_train=25
n_test=200
x_range=(-1.0, 1.0)
noise_std=0.2
seed=0

rng = np.random.default_rng(42)
Xtr = rng.uniform(x_range[0], x_range[1], size=(n_train, 1))
Xte = np.linspace(x_range[0], x_range[1], n_test).reshape(-1, 1)
ytr = model_fn(Xtr) + rng.normal(0.0, noise_std, size=(n_train, 1))
yte = model_fn(Xte)

print(Xtr)