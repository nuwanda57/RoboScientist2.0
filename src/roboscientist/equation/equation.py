import roboscientist.equation.operators as rs_operators

from collections import deque
import numpy as np


class ConstantsCountError(Exception):
    pass


class InvalidEquationError(Exception):
    pass


class Equation:
    def __init__(self, prefix_list):
        self._prefix_list = prefix_list
        self._repr = None
        self._const_count = None
        self._status = self.validate()

    def check_validity(self):
        return self._status == 'OK', self._status

    def repr(self, constants=None):
        if constants is None:
            return self._repr
        stack = deque()
        const_ind = 0
        for elem in self._prefix_list[::-1]:
            if elem in rs_operators.VARIABLES:
                stack.append(elem)
                continue
            if elem == rs_operators.CONST_SYMBOL:
                if constants is not None and const_ind < len(constants):
                    stack.append(str(constants[const_ind]))
                    const_ind += 1
                else:
                    raise ConstantsCountError(f'not enough constants passed {self._prefix_list}, {constants}')
                continue
            if elem in rs_operators.OPERATORS:
                operator = rs_operators.OPERATORS[elem]
                if len(stack) < operator.arity:
                    return f'Invalid Equation {self._prefix_list}'
                args = [stack.pop() for _ in range(operator.arity)]
                stack.append(operator.repr(*args))
                continue
            return f'Invalid symbol in Equation {self._prefix_list}'
        if len(stack) != 1:
            return f'Invalid Equation {self._prefix_list}'
        return stack.pop()

    def const_count(self):
        return self._const_count

    def func(self, X, constants=None):
        stack = deque()
        const_ind = 0
        for elem in self._prefix_list[::-1]:
            if elem in rs_operators.VARIABLES:
                stack.append(X[:,rs_operators.VARIABLES[elem]])
                continue
            if elem == rs_operators.CONST_SYMBOL:
                if constants is not None and const_ind < len(constants):
                    stack.append(constants[const_ind])
                    const_ind += 1
                else:
                    raise ConstantsCountError(f'not enough constants passed {self._prefix_list}, {constants}')
                continue
            if elem in rs_operators.OPERATORS:
                operator = rs_operators.OPERATORS[elem]
                if len(stack) < operator.arity:
                    raise InvalidEquationError(f'Invalid Equation {self._prefix_list}')
                args = [stack.pop() for _ in range(operator.arity)]
                stack.append(operator.func(*args))
                continue
            raise InvalidEquationError(f'Invalid symbol in Equation {self._prefix_list}')
        if len(stack) != 1:
            raise InvalidEquationError(f'Invalid Equation {self._prefix_list}')
        return stack.pop()

    def validate(self):
        self._const_count = 0
        stack = deque()
        for elem in self._prefix_list[::-1]:
            if elem in rs_operators.VARIABLES or elem == rs_operators.CONST_SYMBOL:
                stack.append(elem)
                if elem == rs_operators.CONST_SYMBOL:
                    self._const_count += 1
                continue
            if elem in rs_operators.OPERATORS:
                operator = rs_operators.OPERATORS[elem]
                if len(stack) < operator.arity:
                    return f'Invalid Equation {self._prefix_list}'
                args = [stack.pop() for _ in range(operator.arity)]
                stack.append(operator.repr(*args))
                continue
            return f'Invalid symbol in Equation {self._prefix_list}'
        if len(stack) != 1:
            return f'Invalid Equation {self._prefix_list}'
        self._repr = stack.pop()
        return 'OK'
