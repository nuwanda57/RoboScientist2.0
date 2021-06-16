# TODO(julia): remove this file

from scipy.optimize import minimize

import numpy as np


def _loss(constants, X, y, equation):
    y_hat = equation.func(X, constants)
    loss = (np.real((y_hat - y) ** 2)).mean()
    return np.abs(loss)


def optimize_constants(candidate_equation, X, y):
    if candidate_equation.constants:
        try:
            constants = minimize(lambda constants: _loss(constants, X, y, candidate_equation),
                np.random.uniform(low=0.1, high=1, size=candidate_equation.const_count())).x
            return constants
        except:
            return None
    return None
