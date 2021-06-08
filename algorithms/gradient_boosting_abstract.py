from numpy import random
from pandas import DataFrame

from .Tree import node_based_feature_importance


class GradientBoostingMachine:
    def __init__(self, base_tree,
                 n_estimators,
                 learning_rate,
                 min_samples_leaf,
                 max_depth,
                 min_impurity_decrease,
                 min_samples_split,
                 subsample,
                 bin_numeric_values):
        self.tree = base_tree
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = min_samples_split
        self.n_trees = 0
        self.base_prediction = None
        self.features = None
        self.trees = []
        self.subsample = subsample
        self.bin_numeric_values = bin_numeric_values

    def line_search(self, x, y):
        raise NotImplementedError

    def fit_tree(self, x, y):
        """x: features, y: pseudo response"""
        n_rows = x.shape[0]
        n_samples = int(self.subsample*n_rows)
        # todo: change replace to true???
        indices = random.choice(n_rows, n_samples, replace = False)
        if n_samples == n_rows:
            x_tree = x.copy()
            y_tree = y.copy()
        else:
            x_tree = x.iloc[indices]
            y_tree = y.iloc[indices]
        tree = self.tree(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            min_impurity_decrease=self.min_impurity_decrease,
            min_samples_split=self.min_samples_split)
        tree.fit(x_tree, y_tree, self.bin_numeric_values)
        predictions = tree.predict(x, is_binned = self.bin_numeric_values)
        # TODO: assumption: all leaves provide unique value
        self.trees.append(tree)
        print(tree.n_leaves)
        return predictions

    def fit(self, data: DataFrame):
        raise NotImplementedError

    def predict(self, data: DataFrame):
        raise NotImplementedError

    def compute_feature_importance(self, method='gain'):
        gbm_feature_importances = {feature: 0 for feature in self.features}
        # TODO : deal with the case that a tree is a bark
        for tree in self.trees:
            tree_feature_importance = node_based_feature_importance(tree, method=method)
            for feature, feature_importance in tree_feature_importance.items():
                gbm_feature_importances[feature] += feature_importance
        return gbm_feature_importances
