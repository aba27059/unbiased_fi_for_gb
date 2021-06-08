from numpy import mean, array, log, exp, zeros, ones, unique
from pandas import DataFrame

from . import TreeVisualizer
from .Tree import Leaf, FastCartRegressionTreeKFold, FastCartRegressionTree, CartRegressionTree, \
    CartRegressionTreeKFold, MIN_SAMPLES_LEAF, MAX_DEPTH, MIN_IMPURITY_DECREASE, MIN_SAMPLES_SPLIT
from .config import N_ESTIMATORS, LEARNING_RATE
from .gradient_boosting_abstract import GradientBoostingMachine

SUBSAMPLE = 1.


def tree_n_unique_predictions(tree):
    return len(unique(list(tree.classification_predictions.keys())))


class GradientBoostingClassifier(GradientBoostingMachine):
    """currently supports only binomial log likelihood as in the original paper of friedman"""

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

        self.predictions_to_step_size_dicts = []

    def line_search(self, x, y):
        """
        x: tree predictions: the gradients the tree predicted
        y : pseudo_response: the true gradients
        """
        n_rows = x.shape[0]
        y_ = y.values
        predictions_to_step_size_dict = {}
        for index in range(n_rows):
            predictions_to_step_size_dict.setdefault(x[index], array([0., 0.]))
            predictions_to_step_size_dict[x[index]] += array([y_[index], abs(y_[index]) * (2 - abs(y_[index]))])
        for key, value in predictions_to_step_size_dict.items():
            predictions_to_step_size_dict[key] = value[0] / value[1]
        self.predictions_to_step_size_dicts.append(predictions_to_step_size_dict)
        gamma = zeros(n_rows)
        for index in range(n_rows):
            gamma[index] = predictions_to_step_size_dict[x[index]]

        # if len(predictions_to_step_size_dict) != tree_n_unique_predictions(self.trees[-1]):
        #     print(len(predictions_to_step_size_dict),self.trees[-1].n_leaves)
        #     tree_vis = TreeVisualizer()
        #     print(predictions_to_step_size_dict)
        #     tree_vis.plot(self.trees[-1])
        # assert len(predictions_to_step_size_dict) == tree_n_unique_predictions(self.trees[-1]), "line search map each leaf to a number"
        return gamma

    def fit(self, x, y):
        self.features = x.columns
        y = 2 * y - 1
        self.base_prediction = 0.5 * log((1 + mean(y)) / (1 - mean(y)))
        f = self.base_prediction
        for m in range(self.n_estimators):
            if m > 0 and isinstance(self.trees[-1].root, Leaf):  # if the previous tree was a bark then we stop
                return
            pseudo_response = 2 * y / (1 + exp(2 * y * f))
            h_x = self.fit_tree(x, pseudo_response)
            gamma = self.line_search(h_x, pseudo_response)
            assert len(self.predictions_to_step_size_dicts[-1]) == tree_n_unique_predictions(
                self.trees[-1]), "line search map each leaf to a number"
            f += self.learning_rate * gamma
            self.n_trees += 1

    def predict(self, data: DataFrame):
        prediction = ones(data.shape[0]) * self.base_prediction
        for tree_index, tree in enumerate(self.trees):
            tree_predictions = tree.predict(data, is_binned=self.bin_numeric_values)
            for i in range(tree_predictions.size):
                tree_predictions[i] = self.predictions_to_step_size_dicts[tree_index].get(tree_predictions[i])
            prediction += self.learning_rate * tree_predictions
        return 1 / (1 + exp(-2 * prediction))


class CartGradientBoostingClassifier(GradientBoostingClassifier):
    def __init__(self,
                 n_estimators=N_ESTIMATORS,
                 learning_rate=LEARNING_RATE,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 subsample=SUBSAMPLE):
        super().__init__(
            base_tree=CartRegressionTree,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            subsample=subsample,
        )


class CartGradientBoostingClassifierKfold(GradientBoostingClassifier):
    def __init__(self,
                 n_estimators=N_ESTIMATORS,
                 learning_rate=LEARNING_RATE,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 subsample=SUBSAMPLE):
        super().__init__(
            base_tree=CartRegressionTreeKFold,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            subsample=subsample,
        )


class FastCartGradientBoostingClassifier(GradientBoostingClassifier):
    def __init__(self,
                 n_estimators=N_ESTIMATORS,
                 learning_rate=LEARNING_RATE,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 subsample=SUBSAMPLE):
        super().__init__(
            base_tree=FastCartRegressionTree,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            subsample=subsample,
            bin_numeric_values=True
        )


class FastCartGradientBoostingClassifierKfold(GradientBoostingClassifier):
    def __init__(self,
                 n_estimators=N_ESTIMATORS,
                 learning_rate=LEARNING_RATE,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 subsample=SUBSAMPLE):
        super().__init__(
            base_tree=FastCartRegressionTreeKFold,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            subsample=subsample,
            bin_numeric_values=True
        )
