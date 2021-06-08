import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from algorithms import FastCartGradientBoostingClassifierKfold, \
    FastCartGradientBoostingClassifier, CartGradientBoostingClassifierKfold, CartGradientBoostingClassifier
from algorithms.Tree import TreeVisualizer
from algorithms.Tree.fast_tree.bining import BinMapper
from algorithms.Tree.utils import get_num_cols
from tests.get_xy import get_x_y_breast_cancer, get_x_y_amazon, get_x_y_adult

from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat, get_preprocessing_pipeline
if __name__ == "__main__":

    np.seterr(all='raise')
    KFOLD = True
    FAST = True
    MAX_DEPTH = 3
    np.random.seed(3)

    if FAST:
        model = FastCartGradientBoostingClassifierKfold if KFOLD else FastCartGradientBoostingClassifier
    else:
        model = CartGradientBoostingClassifierKfold if KFOLD else CartGradientBoostingClassifier
    reg = model(max_depth=3)

    X, y = get_x_y_adult()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.1, random_state=42)

    if FAST:
        num_cols = get_num_cols(X.dtypes)
        bin_mapper = BinMapper(max_bins=256, random_state=42)
        X_train.loc[:, num_cols] = bin_mapper.fit_transform(X_train.loc[:, num_cols].values)
        X_test.loc[:, num_cols] = bin_mapper.transform(X_test.loc[:, num_cols].values)

    preprocessing_pipeline = get_preprocessing_pipeline(0.5, [])
    preprocessing_pipeline.fit(X_train)
    X_train = preprocessing_pipeline.transform(X_train)
    X_test = preprocessing_pipeline.transform(X_test)

    start = time.time()
    reg.fit(X_train, y_train)
    end = time.time()
    print(end - start)

    start = time.time()
    print(f"mse is {f1_score(y_test, (reg.predict(X_test) >0.5)*1)}")
    end = time.time()
    print(end - start)

    tree_vis = TreeVisualizer()
    tree_vis.plot(reg.trees[0])

    print(reg.n_trees)
    print(reg.compute_feature_importance())
