# TODO(julia): remove this file

import roboscientist.equation.operators as rs_operators
import roboscientist.equation.equation as rs_equation

from scipy.optimize import minimize

import numpy as np
from copy import deepcopy


def _loss(constants, X, y, equation):
    y_hat = equation.func(X, constants)
    loss = (np.real((y_hat - y) ** 2)).mean()
    return np.abs(loss)


def optimize_constants(candidate_equation, X, y, n_restarts=3):
    best_constants = None
    best_loss = None
    if candidate_equation.const_count() > 0:
        for restart in range(n_restarts):
            res = minimize(lambda constants: _loss(constants, X, y, candidate_equation),
                np.random.uniform(low=0.1, high=1, size=candidate_equation.const_count())).x
            if best_loss is None or res.fun < best_loss:
                best_loss = res.fun
                best_constants = res.x
    return best_constants


# class OperationsTree:
#
#     class Node:
#         def __init__(self, val, parent=None):
#             self.val = val
#             self.kids = []
#             if val in rs_operators.OPERATORS:
#                 self.arity = rs_operators.OPERATORS[val].arity
#             else:
#                 self.arity = 0
#             self.parent = parent
#
#         def __repr__(self):
#             return f'val {self.val}\n, kids {self.kids}\n'
#
#     def build_node(self, node, operations, start_ind):
#         next_ind = start_ind
#         while len(node.kids) != node.arity:
#             new_node = OperationsTree.Node(operations[next_ind], node)
#             node.kids.append(new_node)
#             next_ind = self.build_node(new_node, operations, next_ind + 1)
#         return next_ind
#
#     def __init__(self, operations):
#         self.root = OperationsTree.Node(operations[0])
#         self.build_node(self.root, operations, 1)


# def fill_equation_with_constants(equation):
#
#     new_operations = list()
#     operations = equation.get_prefix_list()
#
#     for i, op in enumerate(operations):
#         new_operations.append(op)
#         if op in {'sin', 'cos'}:
#             new_operations.append('add')
#             new_operations.append(rs_operators.CONST_SYMBOL)
#             new_operations.append('mul')
#             new_operations.append(rs_operators.CONST_SYMBOL)
#         elif op in {'safe_log'}:
#             new_operations.append('add')
#             new_operations.append(rs_operators.CONST_SYMBOL)
#         elif op in {'safe_exp', 'safe_pow'}:
#             new_operations.append('mul')
#             new_operations.append(rs_operators.CONST_SYMBOL)
#         elif op in {'add'}:
#             new_operations.append('mul')
#             new_operations.append(rs_operators.CONST_SYMBOL)
#     new_equation = rs_equation.Equation(new_operations)
#     assert new_equation.check_validity()[0]
#     return new_equation


def fill_equation_with_constants(equation):

    new_operations = list()
    operations = equation.get_prefix_list()

    for i, op in enumerate(operations):
        new_operations.append('add')
        new_operations.append(rs_operators.CONST_SYMBOL)
        new_operations.append('mul')
        new_operations.append(rs_operators.CONST_SYMBOL)
        new_operations.append(op)
    new_equation = rs_equation.Equation(new_operations)
    assert new_equation.check_validity()[0]
    return new_equation


# if __name__ == '__main__':
#     tree = OperationsTree(['mul', 'sin', 'add', 'safe_log', 'x2', 'cos', 'x1', 'safe_sqrt', 'add', 'x1', 'x2'])
#     print(tree.root)