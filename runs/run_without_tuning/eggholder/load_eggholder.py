import numpy as np
from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from scipy.stats import qmc

def load_eggholder_nd(n_samples=1000, n_dims=2, noise=0.0, random_state=None):
    random_state = check_random_state(random_state)
    
    X = random_state.uniform(0, 5, size=(n_samples, n_dims))
    y = np.zeros(n_samples)
    
    for i in range(n_dims - 1):
        xi = X[:, i]
        xj = X[:, i + 1]
        term = -(xj + 47) * np.sin(np.sqrt(np.abs(xi / 2 + xj + 47)) / 5) \
               - xi * np.sin(np.sqrt(np.abs(xi - (xj + 47))) / 5)
        y += term

    y += random_state.normal(scale=noise, size=y.shape)
    
    return shuffle(X, y, random_state=random_state)


def load_eggholder_1D(n_samples=1000, noise=0.0, random_state=None):
    random_state = check_random_state(random_state)
    
    X = np.linspace(0, 20, num=n_samples)
    y = -(X + 47) * np.sin(np.sqrt(np.abs(X + 0.5 * X + 47))) \
        - X * np.sin(np.sqrt(np.abs(X - (X + 47)))) \
        + 5 * np.sin(1.5 * X)
    y += random_state.normal(scale=noise, size=y.shape)
    X = X.reshape(-1, 1)

    return shuffle(X, y, random_state=random_state)

# This eggholder function is smoother and hopefully easier to optimize
def smoothed_eggholder(X):
    n_samples, n_dims = X.shape
    y = np.zeros(n_samples)
    for i in range(n_dims - 1):
        xi = X[:, i]
        xj = X[:, i + 1]
        term = -(xj + 47) * np.sin(np.sqrt(np.abs(xi / 2 + xj + 47)) / 5) \
               - xi * np.sin(np.sqrt(np.abs(xi - (xj + 47))) / 5)
        y += term
    return y

def load_smooth_eggholder_nd(n_samples=1000, n_dims=2, noise=0.0, random_state=None, domain=20):
    random_state = np.random.RandomState(random_state)
    
    if n_dims <= 3 and n_samples <= 10000:
        grid_size = int(np.ceil(n_samples ** (1 / n_dims)))
        linspaces = [np.linspace(0, domain, grid_size) for _ in range(n_dims)]
        mesh = np.meshgrid(*linspaces)
        X = np.vstack([m.ravel() for m in mesh]).T
    else:
        sampler = qmc.LatinHypercube(d=n_dims, seed=random_state)
        X = sampler.random(n_samples) * domain
    
    y = smoothed_eggholder(X)
    
    if noise > 0:
        y += random_state.normal(scale=noise, size=y.shape)
    
    return shuffle(X, y, random_state=random_state)