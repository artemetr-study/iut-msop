import numpy as np


def simplex_max(A: np.array, b: np.array, c: np.array, step_by_step: False) -> dict:
    m, n = A.shape[0], A.shape[1]  # quantity rows and columns of A matrix
    step_by_step_msg = ''

    assert b.size == m, \
        'Quantity elements of b matrix not equal quantity rows of A matrix'

    step_by_step_msg += 'Input data: \n'
    step_by_step_msg += f'A\n{str(A)}\n'
    step_by_step_msg += f'b\n{str(b)}\n'
    step_by_step_msg += f'c\n{str(c)}\n'

    for i in range(m):
        if b[i] >= 0:
            continue
        A[i] *= -1
        b[i] *= -1

    step_by_step_msg += 'If b matrix has one or more negative elements ' \
                        'multiply these elements and related rows in A matrix -1.\n'
    step_by_step_msg += f'A\n{str(A)}\n'
    step_by_step_msg += f'b\n{str(b)}\n'

    step_by_step_msg += 'Building simplex table...\n'
    step_by_step_msg += f'Adding to A matrix identity matrix {m} size.\n'

    b_np = np.array([[i] for i in b] + [[0]], float)
    assert b_np.size == m + 1, \
        'Quantity elements of b_mp matrix not equal quantity elements of b matrix plus 1'

    assert c.size == n, \
        'Quantity elements of c matrix not equal quantity columns of A matrix'

    c_np = np.array([[-i for i in c] + list(np.zeros(m))], float)
    assert c_np.shape[0] == 1 and c_np.shape[1] == n + m, \
        'Invalid size of c_np matrix'

    A_np = np.concatenate((np.concatenate((np.concatenate((A, np.identity(m)), axis=1), c_np), axis=0), b_np), axis=1)
    # If A_np has negative value in latest b column, multiplies row with this value -1.
    bases = np.array(list(range(n, n + m)))

    assert bases.size == m, \
        'Incorrect starting bases set'

    iteration = 0
    optimal_solution_found = False
    problem_msg = 'Too much iteration.'
    while iteration < 500:
        step_by_step_msg += f'\nStep {iteration}\n'
        step_by_step_msg += f'Bases\n{str(bases)}\n'
        step_by_step_msg += f'Simplex table\n{str(A_np)}\n'

        nonzero_b = [i for i in A_np[:, -1] if i != 0]
        if not len(nonzero_b):
            problem_msg = 'B column hasn\'t nonzero values, therefore task has not solution.'
            break

        if not len([i for i in A_np[-1] if i < 0]):
            if set(bases) & set(range(n, n + A.shape[1] - 1)):
                problem_msg = 'Optimal solution has found, but bases has additional variables, ' \
                              'therefore task has not solution.'
                break
            optimal_solution_found = True
            break

        index_column = np.where(A_np[-1][:-2] == np.min(A_np[-1][:-2]))[0][0]
        b_div = A_np[:, -1][:-1] / A_np[:, index_column][:-1]
        if len([i for i in b_div > 0]) == 0:
            problem_msg = 'B column hasn\'t nonzero and positive values, therefore task has not solution.'
            break

        index_row = np.where(b_div == min([i for i in b_div if i > 0]))[0][0]
        A_np[index_row] /= A_np[index_row][index_column]

        step_by_step_msg += f'Support member has coordinates [{index_row} {index_column}]\n'

        for i in range(A_np.shape[0]):
            if i == index_row:
                continue
            A_np[i] -= A_np[index_row] * A_np[i][index_column] / A_np[index_row][index_column]

        bases[index_row] = index_column

        iteration += 1

    if optimal_solution_found:
        step_by_step_msg += 'Optimal solution has been found.\n'
        step_by_step_msg += f'Optimal solution: ' \
                            f'{str(np.array([A_np[:, -1][list(bases).index(i)] if i in bases else 0. for i in range(n)]))}\n'
        step_by_step_msg += f'Value of function: {A_np[-1][-1]}\n'
    else:
        step_by_step_msg += f'{problem_msg}\n'

    return {
        'solution_found': optimal_solution_found,
        'optimal_solution': np.array([A_np[:, -1][list(bases).index(i)] if i in bases else 0. for i in range(n)])
        if optimal_solution_found else None,
        'bases': bases if optimal_solution_found else None,
        'value_of_function': A_np[-1][-1] if optimal_solution_found else None,
        'message': problem_msg if not optimal_solution_found else None,
        'step_by_step': step_by_step_msg if step_by_step else None,
    }

A = np.array([[3, 4, 1, 0, 0],
              [-1, 1, 0, 1, 0],
              [3, 2, 1, 1, 1]], float)

b = np.array([12, 1, 3], float)

c = np.array([1, 5, 2, -1, 1], float)

print(simplex_max(A, b, c, True)['step_by_step'])
