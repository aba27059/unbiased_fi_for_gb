from typing import Tuple

from numba import njit, prange
from numpy import zeros, array, uint16, int32, float32


@njit(cache=True)
def fill_col(col, data, indexes):
    for i in range(indexes.size):
        col[i] = data[indexes[i]]


@njit(parallel=True, cache=True)
def split_x_from_indices(X, left_indices, right_indices):
    x_left = zeros((left_indices.size, X.shape[1]), dtype=uint16)
    x_right = zeros((right_indices.size, X.shape[1]), dtype=uint16)
    for col in prange(X.shape[1]):
        fill_col(x_left[:, col], X[:, col], left_indices)
        fill_col(x_right[:, col], X[:, col], right_indices)

    return x_left, x_right


@njit(cache=True)
def split_y_from_indices(y, left_indices, right_indices):
    y_left = zeros(left_indices.size, dtype=float32)
    y_right = zeros(right_indices.size, dtype=float32)
    fill_col(y_left, y, left_indices)
    fill_col(y_right, y, right_indices)
    return y_left, y_right


@njit(cache=True)
def split_x_y_grad_numeric(col: array, thr: int) -> Tuple[array, array]:
    left_indices = zeros(col.size, dtype=int32)
    right_indices = zeros(col.size, dtype=int32)
    left_index = 0
    right_index = 0
    for val in col:
        index_in_col = left_index + right_index
        if val < thr:
            left_indices[left_index] = index_in_col
            left_index += 1
        else:
            right_indices[right_index] = index_in_col
            right_index += 1
    return left_indices[:left_index], right_indices[: right_index]


@njit(cache=True)
def split_x_y_grad_cat(col: array, left_values) -> Tuple[array, array]:
    left_indices = zeros(col.size, dtype=int32)
    right_indices = zeros(col.size, dtype=int32)
    left_index = 0
    right_index = 0
    for val in col:
        index_in_col = left_index + right_index
        if val in left_values:
            left_indices[left_index] = index_in_col
            left_index += 1
        else:
            right_indices[right_index] = index_in_col
            right_index += 1
    return left_indices[:left_index], right_indices[: right_index]


