import numpy as np
from sklearn.utils import check_random_state, shuffle


def load_linear_nd(n_samples=1000, n_dims=3, weights=None, intercept=0.0, noise=0.0, random_state=None):
    random_state = check_random_state(random_state)

    X = random_state.uniform(low=0.0, high=10.0, size=(n_samples, n_dims))

    if weights is None:
        weights = random_state.uniform(low=-2.0, high=2.0, size=n_dims)
    
    y = X @ weights + intercept
    y += random_state.normal(scale=noise, size=y.shape)

    return shuffle(X, y, random_state=random_state)