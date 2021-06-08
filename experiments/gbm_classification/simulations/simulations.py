from functools import partial
from pathlib import Path

from numpy import random, zeros, arange
from pandas import DataFrame, Series

from algorithms.Tree.utils import get_num_cols
from experiments.config_object import Config
from experiments.default_config import GBM_REGRESSORS, GBM_CLASSIFIERS, SEED, KFOLDS, N_EXPERIMENTS
from experiments.experiment_configurator import experiment_configurator
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat, get_preprocessing_pipeline


def get_x_y(alpha=0.2, null=True):
    if null:
        alpha = 0
    nrows = 6000
    X = DataFrame()
    X['X0'] = random.normal(size = nrows)
    X['X1'] = random.randint(0, 10, nrows)
    X['X2'] = random.randint(0, 20, nrows)
    X['X3'] = random.randint(0, 50, nrows)
    X['X4'] = random.randint(0, 100, nrows)
    for col in X.columns[1:]:
        X[col] = X[col].astype('category')
    y = []
    for i in range(nrows):
        if X['X1'][i] <= 4:
            y.append(random.binomial(1, 0.5 + alpha))
        else:
            y.append(random.binomial(1, 0.5 - alpha))
    y = Series(y)
    return X, y


if __name__ == '__main__':
    MULTIPLE_EXPERIMENTS = True
    KFOLD = False
    ONE_HOT = False
    COMPUTE_PERMUTATION = True
    RESULTS_DIR = Path("null/")

    REGRESSION = False
    x, y = get_x_y()
    contains_num_features = len(get_num_cols(x.dtypes)) > 0
    pp = get_preprocessing_pipeline if contains_num_features else get_preprocessing_pipeline_only_cat
    predictors = GBM_REGRESSORS if REGRESSION else GBM_CLASSIFIERS

    # for results_dir, fun in {Path("null/"): partial(get_x_y, 0, True), Path("0.2/"): partial(get_x_y, 0.2, False)}.items():
    #     config = Config(
    #         multiple_experimens=MULTIPLE_EXPERIMENTS,
    #         n_experiments=100,
    #         kfold_flag=KFOLD,
    #         compute_permutation=COMPUTE_PERMUTATION,
    #         save_results=True,
    #         one_hot=ONE_HOT,
    #         contains_num_features=contains_num_features,
    #         seed=SEED,
    #         kfolds=KFOLDS,
    #         predictors=predictors,
    #         columns_to_remove=[],
    #         get_x_y=fun,
    #         results_dir=results_dir,
    #         preprocessing_pipeline=pp)
    #     experiment_configurator(config)

    for a in arange(0.,0.5,0.05):
        results_dir = Path(F"a_is_changing/{a}")
        fun = partial(get_x_y, a, False)
        config = Config(
            multiple_experimens=MULTIPLE_EXPERIMENTS,
            n_experiments=10,
            kfold_flag=KFOLD,
            compute_permutation=COMPUTE_PERMUTATION,
            save_results=True,
            one_hot=ONE_HOT,
            contains_num_features=contains_num_features,
            seed=SEED,
            kfolds=KFOLDS,
            predictors=predictors,
            columns_to_remove=[],
            get_x_y=fun,
            results_dir=results_dir,
            preprocessing_pipeline=pp)
        experiment_configurator(config)
