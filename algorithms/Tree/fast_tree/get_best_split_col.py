from numba import njit, types, prange
from numba.typed import Dict
from numpy import argsort, array, int32, zeros, arange, random, square

from .gradients import HISTOGRAM_DTYPE, _compute_grad_count_and_sum, grad_subtract, compute_hist_sum

"""
gets the best split for each col. regular or kfold
returns [validation_purity, purity, split_index] and left_values if cat col
"""
KFOLDS= 5


@njit(cache=True)
def pad(a, size):
    l = zeros(size, dtype=int32)
    for index in range(a.size):
        l[index] = a[index]
    return l


@njit(cache=True)
def _get_numeric_node(splitter, X, grad, grad_sums, y, is_regression):
    split_index, purity, _, __ = splitter(grad, grad_sums)
    return array([purity, purity, split_index])


@njit(cache=True)
def _get_categorical_node(splitter, X, grad, grad_sums, y, cat_n_bins, is_regression):
    sorted_indices = argsort(grad['sum_gradients'] / grad['count'])
    split_index, purity, _, __ = splitter(grad[sorted_indices], grad_sums)
    left_values = sorted_indices[:split_index]
    left_values_len = left_values.size
    padded_left_values = pad(left_values, cat_n_bins)
    return array([purity, purity]), padded_left_values, left_values_len


@njit(cache=True)
def get_random_permutation(n):
    arr = arange(n)
    random.shuffle(arr)
    return arr


@njit(cache=True)
def compute_numeric_validation_error(X, y, split_index, left_mean, right_mean, is_regression):
    error = 0
    left_n = 0
    right_n = 0
    left_n_success = 0
    right_n_success = 0
    for row in range(X.size):
        if X[row] <= split_index:
            error += square(y[row] - left_mean)
            left_n += 1
            left_n_success += y[row]
        else:
            error += square(y[row] - right_mean)
            right_n += 1
            right_n_success += y[row]
    if is_regression:
        return error
    else:
        left_p = left_n_success / left_n
        right_p = right_n_success / right_n
        return left_n * left_p * (1 - left_p) + right_n * right_p * (1 - right_p)


@njit(cache=True)
def compute_categorical_validation_error(X, y, left_values, left_mean, right_mean, is_regression):
    error = 0
    left_n = 0
    right_n = 0
    left_n_success = 0
    right_n_success = 0
    for row in prange(X.size):
        if X[row] in left_values:
            error += square(y[row] - left_mean)
            left_n += 1
            left_n_success += y[row]

        else:
            error += square(y[row] - right_mean)
            right_n += 1
            right_n_success += y[row]
    if is_regression:
        return error
    else:
        left_p = left_n_success / left_n
        right_p = right_n_success / right_n
        return left_n * left_p * (1 - left_p) + right_n * right_p * (1 - right_p)


@njit
def _get_numeric_node_kfold(splitter, X, grad, grad_sums, y, is_regression):
    # validation_purity, purity
    split_index, purity, _, __ = splitter(grad, grad_sums)
    return_arr = array([purity, purity, split_index])
    n_rows = X.size
    validation_error = 0
    random_permutation = get_random_permutation(n_rows)
    validation_n_rows = n_rows // KFOLDS
    for i in prange(KFOLDS):
        if i != KFOLDS -1:
            validation_indices = random_permutation[i*validation_n_rows:(i +1)*validation_n_rows]
        else:
            validation_indices = random_permutation[i*validation_n_rows:]
        validation_grad = zeros(256, dtype=HISTOGRAM_DTYPE)
        _compute_grad_count_and_sum(X[validation_indices], y[validation_indices], validation_grad)
        temp_grad = grad.copy()
        grad_subtract(temp_grad, validation_grad)
        temp_grad_sums = compute_hist_sum(temp_grad)
        split_index, _, left_mean, right_mean = splitter(temp_grad, temp_grad_sums)
        validation_error += compute_numeric_validation_error(X[validation_indices], y[validation_indices],
                                                             split_index, left_mean, right_mean, is_regression)
    return_arr[0] = validation_error
    return return_arr


@njit
def _get_categorical_node_kfold(splitter, X, grad, grad_sums, y, cat_n_bins, is_regression):
    # validation_purity, purity
    purity_and_indices, left_values, left_values_len = _get_categorical_node(splitter, X, grad, grad_sums, y,
                                                                             cat_n_bins, is_regression)
    n_rows = X.size
    validation_error = 0
    random_permutation = get_random_permutation(n_rows)
    validation_n_rows = n_rows // KFOLDS
    for i in prange(KFOLDS):
        if i != KFOLDS -1:
            validation_indices = random_permutation[i*validation_n_rows:(i +1)*validation_n_rows]
        else:
            validation_indices = random_permutation[i*validation_n_rows:]
        validation_grad = zeros(cat_n_bins, dtype=HISTOGRAM_DTYPE)
        _compute_grad_count_and_sum(X[validation_indices], y[validation_indices], validation_grad)
        temp_grad = grad.copy()
        grad_subtract(temp_grad, validation_grad)
        temp_grad_sums = compute_hist_sum(temp_grad)
        sorted_indices = argsort(temp_grad['sum_gradients'] / temp_grad['count'])
        split_index, _, left_mean, right_mean = splitter(temp_grad[sorted_indices], temp_grad_sums)
        temp_left_values = sorted_indices[:split_index]
        temp_left_values_dict = Dict.empty(
            key_type=types.int64,
            value_type=types.int8)
        for i in list(temp_left_values):
            temp_left_values_dict[i] = 0
        validation_error += compute_categorical_validation_error(X[validation_indices], y[validation_indices],
                                                                 temp_left_values_dict, left_mean, right_mean,
                                                                 is_regression)
    purity_and_indices[0] = validation_error
    return purity_and_indices, left_values, left_values_len