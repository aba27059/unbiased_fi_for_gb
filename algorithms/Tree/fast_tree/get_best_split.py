from numba import njit, prange
from numpy import argmin, float64, zeros, inf, int64

from .gradients import compute_hist_sum


def get_best_split(X_C, X_N, X_C_G, X_N_G, y, num_node_getter, cat_node_getter, splitter, cat_n_bins, is_regression):
    num_validation_purity, cat_validation_purity = inf, inf
    col_sums = compute_hist_sum(X_C_G[:, 0] if X_C is not None else X_N_G[:, 0])
    if X_N is not None:
        num_best_col, num_validation_purity, num_purity, num_split_index = get_n_best_col(num_node_getter, splitter,
                                                                                          X_N, X_N_G,
                                                                                          col_sums, y, is_regression)
    if X_C is not None:
        cat_best_col, cat_validation_purity, cat_purity, left_values = get_c_best_col(cat_node_getter, splitter, X_C,
                                                                                      X_C_G, col_sums, y, cat_n_bins.astype(int64), is_regression)

    # lower is better
    if num_validation_purity < cat_validation_purity or X_C is None:
        return num_best_col, num_validation_purity, num_purity, int(num_split_index)
    return cat_best_col, cat_validation_purity, cat_purity, set(left_values)


@njit(parallel=True)
def get_n_best_col(num_node_getter, splitter, X_N, X_N_G, col_sums, y, is_regression):
    purity_and_indices = zeros((X_N.shape[1], 3), dtype=float64)
    for col in prange(X_N.shape[1]):
        purity_and_indices[col, :] = num_node_getter(splitter=splitter,
                                                     X=X_N[:, col],
                                                     grad=X_N_G[:, col],
                                                     grad_sums=col_sums, y=y, is_regression = is_regression)
    num_best_col = argmin(purity_and_indices[:, 0])
    num_validation_purity, num_purity, num_split_index = purity_and_indices[num_best_col, :]
    return num_best_col, num_validation_purity, num_purity, num_split_index


@njit(parallel = True)
def get_c_best_col(cat_node_getter, splitter, X_C, X_C_G, col_sums, y, cat_n_bins, is_regression):
    purity_array = zeros((X_C.shape[1], 2), dtype=float64)
    left_values_per_col = zeros((X_C.shape[1], cat_n_bins), dtype=int64)
    left_indices_terminal = zeros(X_C.shape[1], dtype=int64)
    for col in prange(X_C.shape[1]):
        purity_array[col, :], left_values_per_col[col, :], left_indices_terminal[col] \
            = cat_node_getter(splitter=splitter,
                              X=X_C[:, col],
                              grad=X_C_G[:, col],
                              grad_sums=col_sums, y=y, cat_n_bins=cat_n_bins, is_regression = is_regression)

    cat_best_col = argmin(purity_array[:, 0])
    cat_validation_purity, cat_purity = purity_array[cat_best_col, :]
    left_values = left_values_per_col[cat_best_col, :][:left_indices_terminal[cat_best_col]]
    return cat_best_col, cat_validation_purity, cat_purity, left_values
