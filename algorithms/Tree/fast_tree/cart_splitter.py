import numpy as np
from numba import njit


# todo : add if right count == 0: break
@njit(cache=True)
def regression_get_split(grad, grad_counts):
    left_sum, left_counts, left_sum_squares = 0., 0., 0.
    return_left_mean, return_right_mean = 0., 0.
    split_index, best_impurity = -1, np.inf
    n_samples = grad_counts[2]
    _sum = grad_counts[0]
    sum_squares = grad_counts[1]

    for i in range(1, grad.size):
        left_sum += grad[i - 1]['sum_gradients']
        left_sum_squares += grad[i - 1]['sum_squared_gradients']
        left_counts += grad[i - 1]['count']
        if min(left_counts, (n_samples - left_counts)) < 5:
            continue
        left_mean = left_sum / left_counts
        right_mean = (_sum - left_sum) / (n_samples - left_counts)
        left_var = left_sum_squares - left_counts * np.square(left_mean)
        right_var = (sum_squares - left_sum_squares) - (n_samples - left_counts) * np.square(right_mean)
        impurity = left_var + right_var
        if impurity < best_impurity:
            best_impurity, split_index = impurity, i
            return_left_mean, return_right_mean = left_mean, right_mean
    return split_index, best_impurity, return_left_mean, return_right_mean


@njit(cache=True)
def classification_get_split(grad, grad_counts):
    # we can look at left sum for example as number of success in the left split
    left_sum, left_counts = 0., 0.
    return_left_p, return_right_p = 0., 0.
    split_index, best_impurity = -1, np.inf
    n_samples = grad_counts[2]
    _sum = grad_counts[0]

    for i in range(1, grad.size):
        left_sum += grad[i - 1]['sum_gradients']
        left_counts += grad[i - 1]['count']
        if min(left_counts, (n_samples - left_counts)) < 5:
            continue
        left_p = (left_sum / left_counts)
        left_var = left_counts * left_p * (1 - left_p)
        right_p = (_sum - left_sum) / (n_samples - left_counts)
        right_var = (n_samples - left_counts) * right_p * (1 - right_p)
        impurity = left_var + right_var
        if impurity < best_impurity:
            best_impurity, split_index = impurity, i
            return_left_mean, return_right_mean = left_p, right_p
    return split_index, best_impurity, return_left_p, return_right_p
