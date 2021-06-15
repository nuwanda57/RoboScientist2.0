import roboscientist.equation.equation as rs_equation

import numpy as np


if __name__ == '__main__':
    eq = rs_equation.Equation(['add', 'sin', 'x1', 'x1'])
    assert eq.check_validity()[0]
    assert eq.repr() == '(sin(x1) + x1)'
    assert np.allclose(eq.func(X = np.array([[1], [0]])), np.array([1.84147098, 0.]))

    eq = rs_equation.Equation(['add', 'add', 'x1', 'x1', 'x1'])
    assert eq.check_validity()[0]
    assert eq.repr() == '((x1 + x1) + x1)'
    assert np.allclose(eq.func(X=np.array([[1], [0]])), np.array([3., 0.]))

    eq = rs_equation.Equation(['add', 'add', 'x1', 'x2', 'x1'])
    assert eq.check_validity()[0]
    assert eq.repr() == '((x1 + x2) + x1)'
    assert np.allclose(eq.func(X=np.array([[1., 3], [0, 5]])), np.array([5., 5.]))
