from numba import njit, prange
from numpy import square, array, float32, uint32, zeros, dtype

HISTOGRAM_DTYPE = dtype([
    ('sum_gradients', float32),
    ('sum_squared_gradients', float32),
    ('count', uint32),
])

# FEATURES_DTYPE = dtype([
#     ('column_sum_gradients', float32),
#     ('column_sum_squared_gradients', float32),
#     ('column_samples_count', uint32),
# ])


@njit(cache=True)
def grad_subtract(a, b):
    for i in range(a.size):
        a[i]['sum_gradients'] -= b[i]['sum_gradients']
        a[i]['sum_squared_gradients'] -= b[i]['sum_squared_gradients']
        a[i]['count'] -= b[i]['count']


@njit(parallel=True, cache=True)
def subtract_grad(grad_a, grad_b):
    for col in prange(grad_a.shape[1]):
        grad_subtract(grad_a[:, col], grad_b[:, col])
    return grad_a


def compute_children_grad(y_left, y_right, x_left, x_right, node_grad,
                          n_bins):
    if y_left.size < y_right.size:
        x_g_left = compute_grad_sum(x_left, y_left, n_bins)
        x_g_right = subtract_grad(node_grad, x_g_left)
    else:
        x_g_right = compute_grad_sum(x_right, y_right, n_bins)
        x_g_left = subtract_grad(node_grad, x_g_right)
    return x_g_left, x_g_right


@njit(cache=True)
def compute_hist_sum(hist):
    column_sum_gradients = 0
    column_sum_squared_gradients = 0
    column_samples_count = 0
    for i in range(hist.size):
        column_sum_gradients += hist[i]['sum_gradients']
        column_sum_squared_gradients += hist[i]['sum_squared_gradients']
        column_samples_count += hist[i]['count']
    # sums = zeros(3, FEATURES_DTYPE)
    # sums[0]['column_sum_gradients'] = column_sum_gradients
    # sums[0]['column_sum_squared_gradients'] = column_sum_squared_gradients
    # sums[0]['column_samples_count'] = column_samples_count
    sums = zeros(3)
    sums[0] = column_sum_gradients
    sums[1] = column_sum_squared_gradients
    sums[2] = column_samples_count
    return sums


@njit(cache=True)
def _compute_grad_count_and_sum(x, y, hist):
    row = 0
    total_sum = 0
    total_sum_squares = 0
    for val in x:
        grad = y[row]
        grad_square = square(grad)
        hist[val]['sum_gradients'] += grad
        hist[val]['sum_squared_gradients'] += grad_square
        hist[val]['count'] += 1
        total_sum += grad
        total_sum_squares += grad_square
        row += 1
    return hist


@njit(parallel=False)
def compute_grad_sum(X: array, y: array, n_bins):
    """
    create a histogram of gradients
    first histogram : (256/max(unique_values_per_col) as rows ,n_features as cols dtype )
    """
    histogram_n_cols = X.shape[1]
    row_histogram_data = zeros((n_bins, histogram_n_cols), dtype=HISTOGRAM_DTYPE)
    for feature_idx in prange(histogram_n_cols):
        _compute_grad_count_and_sum(*(X[:, feature_idx], y, row_histogram_data[:, feature_idx]))
    return row_histogram_data

