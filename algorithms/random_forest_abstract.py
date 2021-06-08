import concurrent.futures

from numpy import random, sqrt, zeros

from algorithms import node_based_feature_importance


def get_fitted_tree(self, x, y):
    tree = self.base_tree(min_samples_leaf=self.min_samples_leaf,
                          max_depth=self.max_depth,
                          min_impurity_decrease=self.min_impurity_decrease,
                          min_samples_split=self.min_samples_split)
    features = random.choice(self.n_cols, self.n_features, replace=False)
    replace = True if self.bootstrap else False
    rows = random.choice(self.n_rows, self.nrows_per_tree, replace=replace)
    temp_x, temp_y = x.iloc[rows, features], y.iloc[rows]
    tree.fit(temp_x, temp_y)
    return tree


class RandomForest:
    def __init__(self, base_tree,
                 n_features,
                 bootstrap,
                 n_estimators,
                 min_samples_leaf,
                 max_depth,
                 min_impurity_decrease,
                 min_samples_split,
                 max_samples_fraction,
                 random_state):
        self.base_tree = base_tree
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = min_samples_split
        self.max_samples_fraction = max_samples_fraction
        self.random_state = random_state
        self.n_rows, self.n_cols, self.nrows_per_tree = None, None, None
        self.trees = []
        self.features = None

    def fit(self, X, y):
        self.features = X.columns
        self.n_rows, self.n_cols = X.shape
        self.nrows_per_tree = int(self.n_rows * self.max_samples_fraction)
        if self.random_state:
            random.seed(self.random_state)
        if self.n_features == "sqrt":
            self.n_features = int(sqrt(self.n_cols))

        # with concurrent.futures.ProcessPoolExecutor() as e:
        #     results = [e.submit(get_fitted_tree, self, x, y) for _ in range(self.n_estimators)]
        #     self.trees = [f.result() for f in concurrent.futures.as_completed(results)]
        self.trees = [get_fitted_tree(self, X, y) for _ in range(self.n_estimators)]

    def predict(self, x):
        prediction = zeros(x.shape[0])
        for tree in self.trees:
            prediction += tree.predict(x, is_binned=True)
        return prediction / self.n_estimators
    
    def compute_feature_importance(self, method='gain'):
        rf_feature_importances = {feature: 0 for feature in self.features}
        # TODO : deal with the case that a tree is a bark
        for tree in self.trees:
            tree_feature_importance = node_based_feature_importance(tree, method=method)
            for feature, feature_importance in tree_feature_importance.items():
                rf_feature_importances[feature] += feature_importance
        return rf_feature_importances
