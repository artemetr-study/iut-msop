import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

from fibonachi_method import get_extremum_of_function_by_fibonacci_method
from method_of_bisection import get_extremum_of_function_by_method_of_bisections
from scipy_internal_method import \
    get_extremum_of_function_by_scipy_internal_method


def get_limits_step_and_scale(values: list) -> (list, float):
    min_value = min(values)
    max_value = max(values)
    step = (max_value - min_value) / 0.5

    return [min_value - step, max_value + step], step


def main():
    a = 1.4
    b = 2
    fidelity = pow(10, -2)

    def function(x: float) -> float:
        return 2 * pow(x, 2) + 3 * pow(5 - x, 4 / 3)

    # extremum points
    fibonacci_x = get_extremum_of_function_by_fibonacci_method(
        function, a, b, fidelity)
    bisection_x = get_extremum_of_function_by_method_of_bisections(
        function, a, b, fidelity)
    scipy_x = get_extremum_of_function_by_scipy_internal_method(
        function, a, b, fidelity)

    # make plot
    limits_x, step_x = get_limits_step_and_scale([fibonacci_x,
                                                  bisection_x,
                                                  scipy_x])
    limits_y, step_y = get_limits_step_and_scale([function(x) for x
                                                  in [fibonacci_x,
                                                      bisection_x,
                                                      scipy_x]])

    fig = plt.figure(figsize=(6, 4), dpi=300)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlim(limits_x[0], limits_x[1])
    ax.set_ylim(limits_y[0], limits_y[1])

    ax.xaxis.set_major_locator(MultipleLocator(step_x))
    ax.yaxis.set_major_locator(MultipleLocator(step_y))

    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    ax.grid(which='major', color='#666666', linestyle='--')
    ax.grid(which='minor', color='#DDDDDD', linestyle=':')

    x = np.arange(a, b, fidelity * pow(10, -2))
    ax.plot(x, function(x), label="f(x)", color='#000000', linestyle='-')

    # scipy
    min_x = np.array([scipy_x])
    ax.plot(min_x, function(min_x), 'ko', label="extremum point by scipy "
                                                "internal method")
    # bisection
    min_x = np.array([bisection_x])
    ax.plot(min_x, function(min_x), 'kv',
            label="extremum point by "
                  "method of "
                  "bisections")

    # fibonacci
    min_x = np.array([fibonacci_x])
    ax.plot(min_x, function(min_x), 'ks',
            label="extremum point by fibonacci method")

    # Decorate the figure
    ax.legend(loc='best')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.show()

    dataset = pd.DataFrame({"fibonacci_method": {"x": fibonacci_x,
                                                 "y": function(
                                                     fibonacci_x)},
                            "bisection_method": {"x": bisection_x,
                                                 "y": function(
                                                     bisection_x)},
                            "scipy_method": {"x": scipy_x,
                                             "y": function(
                                                 scipy_x)}
                            })
    print(dataset)


if __name__ == '__main__':
    main()
