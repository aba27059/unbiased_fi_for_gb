from pathlib import Path

from pandas import read_csv, factorize, Series

from algorithms.Tree.utils import get_num_cols
from experiments.config_object import Config
from experiments.default_config import GBM_CLASSIFIERS, GBM_REGRESSORS, N_EXPERIMENTS, SEED, KFOLDS
from experiments.preprocess_pipelines import get_preprocessing_pipeline_only_cat, get_preprocessing_pipeline
from experiments.experiment_configurator import experiment_configurator


def get_x_y():
    project_root = Path(__file__).parent.parent.parent.parent
    y_col_name = 'y'
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
               'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'y']

    train = read_csv(project_root / "datasets/adult/adult.data", header=None)
    train.columns = columns
    y = Series(factorize(train[y_col_name])[0])
    X = train.drop(columns=[y_col_name])
    for col in X.select_dtypes(include=['O']).columns.tolist():
        X[col] = X[col].astype('category')
    return X, y


if __name__ == '__main__':
    MULTIPLE_EXPERIMENTS = True
    KFOLD = True
    ONE_HOT = False
    COMPUTE_PERMUTATION = True
    RESULTS_DIR = Path("30Fold/")

    REGRESSION = False
    x, y = get_x_y()
    contains_num_features = len(get_num_cols(x.dtypes)) > 0
    pp = get_preprocessing_pipeline if contains_num_features else get_preprocessing_pipeline_only_cat
    predictors = GBM_REGRESSORS if REGRESSION else GBM_CLASSIFIERS
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
        get_x_y=get_x_y,
        results_dir=RESULTS_DIR,
        preprocessing_pipeline=pp)
    experiment_configurator(config)
