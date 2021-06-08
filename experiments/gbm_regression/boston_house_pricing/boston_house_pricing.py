from pathlib import Path

from numpy import log
from pandas import DataFrame, Series
from sklearn.datasets import load_boston

from algorithms.Tree.utils import get_num_cols
from experiments.config_object import Config
from experiments.default_config import GBM_CLASSIFIERS, N_EXPERIMENTS, SEED, KFOLDS
from experiments.default_config import GBM_REGRESSORS
from experiments.experiment_configurator import experiment_configurator
from experiments.preprocess_pipelines import get_preprocessing_pipeline
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat

EXP = 'boston_hp'


def get_x_y(apply_log=True):
    data = load_boston()
    X = DataFrame(data['data'], columns=data['feature_names'])
    y = Series(data['target'])
    if apply_log:
        y = log(y)
    return X, y


if __name__ == '__main__':
    MULTIPLE_EXPERIMENTS = False
    KFOLD = True
    ONE_HOT = False
    COMPUTE_PERMUTATION = False
    RESULTS_DIR = Path(F"{EXP}_Log/")

    x, y = get_x_y()
    regression = not (len(y.value_counts()) == 2)
    contains_num_features = len(get_num_cols(x.dtypes)) > 0
    pp = get_preprocessing_pipeline if contains_num_features else get_preprocessing_pipeline_only_cat
    predictors = GBM_REGRESSORS if regression else GBM_CLASSIFIERS

    config = Config(
        multiple_experimens=MULTIPLE_EXPERIMENTS,
        n_experiments=N_EXPERIMENTS,
        kfold_flag=KFOLD,
        compute_permutation=COMPUTE_PERMUTATION,
        save_results=True,
        one_hot=ONE_HOT,
        contains_num_features=contains_num_features,
        seed=SEED,
        kfolds=KFOLDS,
        predictors=predictors,
        columns_to_remove=[],
        get_x_y=get_x_y,
        results_dir=RESULTS_DIR,
        preprocessing_pipeline=pp)
    experiment_configurator(config)
