import numpy as np
from scipy import optimize


def get_extremum_of_function_by_scipy_internal_method(function, a, b, fidelity)\
        ->\
        float:
    """
    Function that returns extremum of function by scipi internal method.
    Args:
        function (function): python function like lambda x: x * 2;
        a (float): starting point of the search interval;
        b (float): ending point of the search interval;
        fidelity(float): fidelity of extremum calculation.
    Returns:
        float: extremum point of function.
    """
    grid = (a, b, fidelity)
    return float(optimize.brute(function, (grid,)))