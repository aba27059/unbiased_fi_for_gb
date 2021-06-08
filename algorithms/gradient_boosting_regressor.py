from numpy import mean, ones
from pandas import DataFrame

from .Tree import CartRegressionTree, CartRegressionTreeKFold, MIN_SAMPLES_LEAF, MAX_DEPTH, MIN_IMPURITY_DECREASE, \
    MIN_SAMPLES_SPLIT
from .Tree import Leaf, FastCartRegressionTreeKFold, FastCartRegressionTree
from .config import N_ESTIMATORS, LEARNING_RATE
from .gradient_boosting_abstract import GradientBoostingMachine

SUBSAMPLE = 1.


class GradientBoostingRegressor(GradientBoostingMachine):
    """currently supports least squares"""

    def __init__(self, base_tree,
                 n_estimators,
                 learning_rate,
                 min_samples_leaf,
                 max_depth,
                 min_impurity_decrease,
                 min_samples_split,
                 subsample,
                 bin_numeric_values=False):
        super().__init__(
            base_tree=base_tree,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            subsample=subsample,
            bin_numeric_values=bin_numeric_values)

    def line_search(self, x, y):
        return x

    def fit(self, X, y):
        self.features = X.columns
        self.base_prediction = mean(y)
        f = mean(y)
        for m in range(self.n_estimators):
            if m > 0 and isinstance(self.trees[-1].root, Leaf):  # if the previous tree was a bark then we stop
                return
            pseudo_response = y - f
            h_x = self.fit_tree(X, pseudo_response)
            gamma = self.line_search(h_x, y)
            f += self.learning_rate * gamma
            self.n_trees += 1

    def worker(self, tree, data, is_binned):
        return tree.predict(data, is_binned)

    def predict(self, data: DataFrame):
        prediction = ones(data.shape[0]) * self.base_prediction
        for tree_index, tree in enumerate(self.trees):
            prediction += self.learning_rate * tree.predict(data, is_binned=self.bin_numeric_values)
        return prediction


class CartGradientBoostingRegressor(GradientBoostingRegressor):
    def __init__(self,
                 n_estimators=N_ESTIMATORS,
                 learning_rate=LEARNING_RATE,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 subsample = SUBSAMPLE):
        super().__init__(
            base_tree=CartRegressionTree,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            subsample=subsample)


class CartGradientBoostingRegressorKfold(GradientBoostingRegressor):
    def __init__(self,
                 n_estimators=N_ESTIMATORS,
                 learning_rate=LEARNING_RATE,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 subsample = SUBSAMPLE):
        super().__init__(
            base_tree=CartRegressionTreeKFold,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            subsample=subsample)


class FastCartGradientBoostingRegressor(GradientBoostingRegressor):
    def __init__(self,
                 n_estimators=N_ESTIMATORS,
                 learning_rate=LEARNING_RATE,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 subsample = SUBSAMPLE):
        super().__init__(
            base_tree=FastCartRegressionTree,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            subsample=subsample,
            bin_numeric_values=True)


class FastCartGradientBoostingRegressorKfold(GradientBoostingRegressor):
    def __init__(self,
                 n_estimators=N_ESTIMATORS,
                 learning_rate=LEARNING_RATE,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 subsample = SUBSAMPLE):
        super().__init__(
            base_tree=FastCartRegressionTreeKFold,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            subsample=subsample,
            bin_numeric_values=True)
