import time
from typing import List

from numba import types
from numba.typed import Dict
from numpy import zeros, array, random, max, uint16, ones, abs, sum
from pandas import DataFrame, Series, concat

from algorithms.Tree.fast_tree.bining import BinMapper
from algorithms.Tree.fast_tree.get_best_split import get_best_split
from algorithms.Tree.fast_tree.get_best_split_col import _get_numeric_node, _get_categorical_node, \
    _get_categorical_node_kfold, \
    _get_numeric_node_kfold
from algorithms.Tree.fast_tree.gradients import compute_grad_sum, compute_children_grad
from algorithms.Tree.fast_tree.split_data import split_x_y_grad_cat, split_x_y_grad_numeric, split_x_from_indices, \
    split_y_from_indices
from algorithms.Tree.fast_tree.cart_splitter import regression_get_split, classification_get_split
from algorithms.Tree.config import MIN_SAMPLES_LEAF, MAX_DEPTH, MIN_IMPURITY_DECREASE, MIN_SAMPLES_SPLIT
from algorithms.Tree.node import InternalNode, Leaf, CategoricalBinaryNode, NumericBinaryNode
from algorithms.Tree.utils import get_cols_dtypes, classification_impurity, regression_impurity, get_cat_num_cols, \
    get_max_value_per_cat


class BaseTree:
    def __init__(self, kfold, regressor, min_samples_leaf,
                 max_depth, min_impurity_decrease, min_samples_split, is_regression=False):
        self.cat_node_getter = _get_categorical_node_kfold if kfold else _get_categorical_node
        self.num_node_getter = _get_numeric_node_kfold if kfold else _get_numeric_node
        self.splitter = regression_get_split if regressor else classification_get_split
        self.min_impurity_decrease = min_impurity_decrease
        self.impurity = regression_impurity if regressor else classification_impurity
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.is_regression = is_regression
        self.root = None
        self.n_leaves = 0
        self.nodes = []
        self.edges = []
        self.children_left = []
        self.children_right = []
        self.features = []
        self.node_impurity = []
        self.n_node_samples = []
        self.bin_mapper = None
        self.n_unique_values_per_cat = None
        self.cat_cols = None
        self.num_cols = None
        self.cat_n_bins = None
        self.column_dtypes = None
        self.classification_predictions = {}

    def calculate_impurity(self, y) -> float:
        return self.impurity(y)

    def _best_split(self, X_C, X_N, X_C_G, X_N_G, y):
        best_col, validation_purity, purity, indicator = \
            get_best_split(X_C, X_N, X_C_G, X_N_G, y, self.num_node_getter, self.cat_node_getter, self.splitter,
                           self.cat_n_bins, self.is_regression)
        if type(indicator) is int:
            node = NumericBinaryNode(
                n_examples=X_N.shape[0],
                split_purity=purity,
                field=self.num_cols[best_col],
                splitting_point=indicator)

            left_indices, right_indices = split_x_y_grad_numeric(X_N[:, best_col], node.thr)
        else:
            node = CategoricalBinaryNode(
                n_examples=X_C.shape[0],
                split_purity=purity,
                field=self.cat_cols[best_col],
                left_values=indicator)
            left_values = Dict.empty(
                key_type=types.int64,
                value_type=types.int8)
            for i in list(node.left_values):
                left_values[i] = 0
            left_indices, right_indices = split_x_y_grad_cat(X_C[:, best_col], left_values)

        return node, validation_purity, left_indices, right_indices

    def _grow_tree(self, X_C: array, X_N: array, X_C_G: array, X_N_G: array, y: array, depth: int) -> [InternalNode,
                                                                                                       Leaf]:
        impurity = self.calculate_impurity(y)
        n_samples = X_C.shape[0] if X_C is not None else X_N.shape[0]
        classification_prediction = sum(y)/sum(abs(y) * (2 - abs(y))) # for gradient boosting trees
        leaf = Leaf(y.mean(), "_", n_samples, impurity, classification_prediction)
        is_leaf = (impurity == 0) or (n_samples <= self.min_samples_split) or (depth == self.max_depth)
        if is_leaf:
            return leaf
        node, node_score, left_indices, right_indices = self._best_split(X_C, X_N, X_C_G, X_N_G, y)
        is_leaf = (node is None) or ((impurity - node_score) < self.min_impurity_decrease)
        if is_leaf:
            return leaf
        node.purity = impurity
        if left_indices.size == 0 or right_indices.size == 0:
            return leaf
        y_left, y_right = split_y_from_indices(y, left_indices, right_indices)
        if X_C is None:
            x_c_left, x_c_right, x_c_g_left, x_c_g_right = None, None, None, None
        else:
            x_c_left, x_c_right = split_x_from_indices(X_C, left_indices, right_indices)
            x_c_g_left, x_c_g_right = compute_children_grad(y_left, y_right, x_c_left,
                                                            x_c_right, X_C_G,
                                                            self.cat_n_bins)
        if X_N is None:
            x_n_left, x_n_right, x_n_g_left, x_n_g_right = None, None, None, None
        else:
            x_n_left, x_n_right = split_x_from_indices(X_N, left_indices, right_indices)
            x_n_g_left, x_n_g_right = compute_children_grad(y_left, y_right, x_n_left,
                                                            x_n_right,
                                                            X_N_G,
                                                            256)

        node.left = self._grow_tree(x_c_left, x_n_left, x_c_g_left, x_n_g_left, y_left, depth + 1)
        node.right = self._grow_tree(x_c_right, x_n_right, x_c_g_right, x_n_g_right, y_right, depth + 1)
        node.add_depth(depth)
        return node

    def fit(self, X: DataFrame, y: Series, is_binned=False):
        self.column_dtypes = get_cols_dtypes(X)
        self.cat_cols, self.num_cols = get_cat_num_cols(X.dtypes)
        X_cat, X_num = None, None
        cat_grad_data, num_grad_data = None, None
        y = y.values
        if self.cat_cols:
            X_cat = X[self.cat_cols].values.astype(uint16)
            self.n_unique_values_per_cat = get_max_value_per_cat(X_cat)
            self.cat_n_bins = max(self.n_unique_values_per_cat) + 1
            cat_grad_data = compute_grad_sum(X_cat, y, self.cat_n_bins)
        if self.num_cols:
            X_num = X[self.num_cols].values
            if not is_binned:
                bin_mapper = BinMapper(max_bins=256, random_state=42)
                X_num = bin_mapper.fit_transform(X_num)
                self.bin_mapper = bin_mapper
            num_grad_data = compute_grad_sum(X_num, y, 256)
        self.root = self._grow_tree(X_cat, X_num, cat_grad_data, num_grad_data, y, 0)
        if isinstance(self.root, Leaf):
            self.n_leaves = 1
            self.classification_predictions[self.root.prediction] = 0
        else:
            self.number_nodes_and_update_tree_data()

    # def predict(self, x: DataFrame, is_binned: False) -> array:
    #     if not is_binned:
    #         X.loc[:, self.num_cols] = self.bin_mapper.transform(X, X.loc[:, self.num_cols])
    #     records = x.to_dict('records')
    #     results = zeros(len(records))
    #     for i, row in enumerate(records):
    #         node = self.root
    #         while isinstance(node, InternalNode):
    #             value = row[node.field]
    #             node = node.get_child(value)
    #         results[i] = node.prediction
    #     return results

    def predict(self, x: DataFrame, is_binned: False) -> array:
        if not is_binned:
            X.loc[:, self.num_cols] = self.bin_mapper.transform(X, X.loc[:, self.num_cols])
        queue = [[(self.root, x)]]
        leaves = []
        if isinstance(self.root, Leaf):
            return ones(x.shape[0]) * self.root.prediction
        while queue:
            level_nodes = queue.pop(0)
            next_level_nodes = []
            for node, node_data in level_nodes:  # only on internal nodes
                if isinstance(node, NumericBinaryNode):
                    # TODO: check if it correct
                    left_indices = node_data[node.field] < node.thr
                else:
                    left_indices = node_data[node.field].apply(lambda x: True if x in node.left_values else False)
                # right_indices = left_indices.apply()
                data_dict = dict(tuple(node_data.groupby(left_indices)))
                for child in [(node.left, True), (node.right, False)]:
                    if data_dict.get(child[1]) is None:
                        continue
                    if isinstance(child[0], Leaf):
                        leaves.append((child[0], data_dict.get(child[1])))
                    else:
                        next_level_nodes.append((child[0], data_dict.get(child[1])))
            if next_level_nodes:
                queue.append(next_level_nodes)
        results = []
        for leaf, leaf_data in leaves:
            results.append(Series(leaf.prediction, leaf_data.index))
        results = concat(results)
        return results[x.index].values

    def number_nodes_and_update_tree_data(self):
        self.root.number = 0
        queue = [[self.root]]
        counter = 0
        while queue:
            level_nodes = queue.pop(0)
            next_level_nodes = []
            for node in level_nodes:
                self.nodes.append(node)
                if isinstance(node, InternalNode):
                    self.features.append(node.field)
                    node.left.number, node.right.number = counter + 1, counter + 2
                    self.edges += [(node.number, node.left.number), (node.number, node.right.number)]
                    next_level_nodes += [node.left, node.right]
                    counter += 2
                    self.children_left.append(node.left.number)
                    self.children_right.append(node.right.number)
                else:
                    self.n_leaves += 1
                    self.features.append(-2)
                    self.children_left.append(-1)
                    self.children_right.append(-1)
                    self.classification_predictions[node.prediction] = node.classification_prediction

                self.node_impurity.append(node.purity)
                self.n_node_samples.append(node.n_examples)

            if next_level_nodes:
                queue.append(next_level_nodes)


class FastCartRegressionTree(BaseTree):
    def __init__(self,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(kfold=False,
                         regressor=True,
                         min_samples_leaf=min_samples_leaf,
                         max_depth=max_depth,
                         min_impurity_decrease=min_impurity_decrease,
                         min_samples_split=min_samples_split,
                         is_regression=True)


class FastCartClassificationTree(BaseTree):
    def __init__(self,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(kfold=False,
                         regressor=False,
                         min_samples_leaf=min_samples_leaf,
                         max_depth=max_depth,
                         min_impurity_decrease=min_impurity_decrease,
                         min_samples_split=min_samples_split)


class FastCartRegressionTreeKFold(BaseTree):
    def __init__(self,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(kfold=True,
                         regressor=True,
                         min_samples_leaf=min_samples_leaf,
                         max_depth=max_depth,
                         min_impurity_decrease=min_impurity_decrease,
                         min_samples_split=min_samples_split,
                         is_regression=True)


class FastCartClassificationTreeKFold(BaseTree):
    def __init__(self,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT):
        super().__init__(kfold=True,
                         regressor=True,
                         min_samples_leaf=min_samples_leaf,
                         max_depth=max_depth,
                         min_impurity_decrease=min_impurity_decrease,
                         min_samples_split=min_samples_split)


def create_x_y(regression=True):
    df = DataFrame()
    n_rows = 10 ** 4
    n_numeric_cols = 10
    n_categorical_cols = 10
    n_categorical_values = 10
    for col in range(n_numeric_cols):
        df[col] = random.random(n_rows)
    for col in range(n_categorical_cols):
        df[col + n_numeric_cols] = random.randint(n_categorical_values, size=n_rows)
        df[col + n_numeric_cols] = random.randint(n_categorical_values, size=n_rows)
        df[col + n_numeric_cols] = df[col + n_numeric_cols].astype('category')
    y = random.random(n_rows) if regression else random.randint(2, size=n_rows)
    return df, Series(y)


if __name__ == "__main__":
    from algorithms.Tree.tree_visualizer import TreeVisualizer

    KFOLD = True
    MAX_DEPTH = 1
    tree = FastCartRegressionTreeKFold(max_depth=MAX_DEPTH) if KFOLD else FastCartRegressionTree(max_depth=MAX_DEPTH)
    random.seed(10)
    X, y = create_x_y()
    start = time.time()
    tree.fit(X, y)
    end = time.time()
    print(end - start)
    tree_vis = TreeVisualizer()
    tree_vis.plot(tree)
