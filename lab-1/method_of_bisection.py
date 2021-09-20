import numpy as np


def get_extremum_of_function_by_method_of_bisections(function, a: float,
                                                     b: float,
                                                     fidelity: float) -> float:
    """
    Function that returns extremum of function by method of bisections.
    Args:
        function (function): python function like lambda x: x * 2;
        a (float): starting point of the search interval;
        b (float): ending point of the search interval;
        fidelity (float): fidelity of extremum calculation.

    Returns:
        float: extremum point of function.
    """
    x_list = np.linspace(a, b, 4).tolist()
    y_list = [function(x) for x in x_list]
    lower_index = np.argmin(y_list)

    if lower_index == 0 or lower_index == 3:
        raise ValueError("Invalid range")

    if x_list[lower_index + 1] - x_list[lower_index - 1] < fidelity:
        if y_list[lower_index - 1] < y_list[lower_index + 1]:
            x_needed = x_list[lower_index - 1]
        else:
            x_needed = x_list[lower_index + 1]
        return x_needed

    return get_extremum_of_function_by_method_of_bisections \
        (function, x_list[lower_index - 1], x_list[lower_index + 1], fidelity)
