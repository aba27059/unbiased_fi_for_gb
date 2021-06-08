from .Tree import CartClassificationTree, CartClassificationTreeKFold, \
    MIN_SAMPLES_LEAF, MAX_DEPTH, MIN_IMPURITY_DECREASE, MIN_SAMPLES_SPLIT, FastCartClassificationTree, \
    FastCartClassificationTreeKFold
from .config import N_FEATURES, BOOTSTRAP, N_ESTIMATORS, MAX_SAMPLES_FRACTION, RANDOM_STATE
from .random_forest_abstract import RandomForest


class CartRandomForestClassifier(RandomForest):
    def __init__(self,
                 n_features=N_FEATURES,
                 bootstrap=BOOTSTRAP,
                 n_estimators=N_ESTIMATORS,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 max_samples_fraction=MAX_SAMPLES_FRACTION,
                 random_state=RANDOM_STATE):
        super().__init__(
            base_tree=CartClassificationTree,
            n_features=n_features,
            bootstrap=bootstrap,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            max_samples_fraction=max_samples_fraction,
            random_state=random_state)


class CartKfoldRandomForestClassifier(RandomForest):
    def __init__(self,
                 n_features=N_FEATURES,
                 bootstrap=BOOTSTRAP,
                 n_estimators=N_ESTIMATORS,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 max_samples_fraction=MAX_SAMPLES_FRACTION,
                 random_state=RANDOM_STATE):
        super().__init__(
            base_tree=CartClassificationTreeKFold,
            n_features=n_features,
            bootstrap=bootstrap,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            max_samples_fraction=max_samples_fraction,
            random_state=random_state)


class FastCartRandomForestClassifier(RandomForest):
    def __init__(self,
                 n_features=N_FEATURES,
                 bootstrap=BOOTSTRAP,
                 n_estimators=N_ESTIMATORS,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 max_samples_fraction=MAX_SAMPLES_FRACTION,
                 random_state=RANDOM_STATE):
        super().__init__(
            base_tree=FastCartClassificationTree,
            n_features=n_features,
            bootstrap=bootstrap,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            max_samples_fraction=max_samples_fraction,
            random_state=random_state)


class FastCartKfoldRandomForestClassifier(RandomForest):
    def __init__(self,
                 n_features=N_FEATURES,
                 bootstrap=BOOTSTRAP,
                 n_estimators=N_ESTIMATORS,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH,
                 min_impurity_decrease=MIN_IMPURITY_DECREASE,
                 min_samples_split=MIN_SAMPLES_SPLIT,
                 max_samples_fraction=MAX_SAMPLES_FRACTION,
                 random_state=RANDOM_STATE):
        super().__init__(
            base_tree=FastCartClassificationTreeKFold,
            n_features=n_features,
            bootstrap=bootstrap,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_split=min_samples_split,
            max_samples_fraction=max_samples_fraction,
            random_state=random_state)
