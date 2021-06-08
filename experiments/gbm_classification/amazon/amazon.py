from functools import partial
from pathlib import Path

from pandas import read_csv

from algorithms.Tree.utils import get_num_cols
from experiments.config_object import Config
from experiments.default_config import GBM_CLASSIFIERS, GBM_REGRESSORS, N_EXPERIMENTS, SEED, KFOLDS
from experiments.experiment_configurator import experiment_configurator
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat, get_preprocessing_pipeline


def get_x_y(with_resource=True):
    project_root = Path(__file__).parent.parent.parent.parent
    y_col_name = 'ACTION'
    train = read_csv(project_root / 'datasets/amazon_from_catboost_paper/train.csv')
    y = train[y_col_name]
    columns_to_drop = [y_col_name]
    if not with_resource:
        columns_to_drop.append('RESOURCE')
    X = train.drop(columns=columns_to_drop)
    for col in X.columns:
        X[col] = X[col].astype('category')
    return X, y


if __name__ == '__main__':
    MULTIPLE_EXPERIMENTS = False
    KFOLD = True
    ONE_HOT = False
    COMPUTE_PERMUTATION = True
    CONTAINS_NUM_FEATURES = False
    RESULTS_DIR = Path("30foldCV_new_no_resource/")

    REGRESSION = False
    x, y = get_x_y()
    contains_num_features = len(get_num_cols(x.dtypes)) > 0
    pp = get_preprocessing_pipeline if CONTAINS_NUM_FEATURES else get_preprocessing_pipeline_only_cat
    predictors = GBM_REGRESSORS if REGRESSION else GBM_CLASSIFIERS

    for results_dir, fun in {Path("30foldCV_no_resource/"): partial(get_x_y, False),
                             Path("30foldCV/"): partial(get_x_y, True)}.items():
        config = Config(
            multiple_experimens=MULTIPLE_EXPERIMENTS,
            n_experiments=N_EXPERIMENTS,
            kfold_flag=KFOLD,
            compute_permutation=COMPUTE_PERMUTATION,
            save_results=True,
            one_hot=ONE_HOT,
            contains_num_features=contains_num_features,
            seed=SEED,
            kfolds=30,
            predictors=predictors,
            columns_to_remove=[],
            get_x_y=fun,
            results_dir=results_dir,
            preprocessing_pipeline=pp)
        experiment_configurator(config)
