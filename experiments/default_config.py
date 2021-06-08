from pathlib import Path
from typing import Dict

from experiments.moodel_wrappers.gbm import *
from experiments.moodel_wrappers.random_forest import SklearnRfRegressorWrapper, SklearnRfClassifierWrapper
# gbm
from experiments.moodel_wrappers.random_forest.our_rf_wrapper import OurFastRfRegressorWrapper, \
    OurFastRfClassifierWrapper, OurFastKfoldRfClassifierWrapper, \
    OurFastKfoldRfRegressorWrapper

MAX_DEPTH = 3
N_ESTIMATORS = 100
LEARNING_RATE = 0.1
SUBSAMPLE = 1

# io
# MODELS_DIR = Path(F"results/models/")
RESULTS_DIR = Path(F"results/")

# data
CATEGORY_COLUMN_NAME = 'category'
VAL_RATIO = 0.15
Y_COL_NAME = 'y'
N_ROWS = 10 ** 3

SEED = 10
KFOLDS = 10
N_EXPERIMENTS = 10


class Models:
    def __init__(self, is_gbm: bool, models_dict: Dict):
        self.is_gbm = is_gbm
        self.models_dict = models_dict


GBM_REGRESSORS = Models(True, {
    'lgbm': LgbmGbmRegressorWrapper,
    'xgboost': XgboostGbmRegressorWrapper,
    'catboost': CatboostGbmRegressorWrapper,
    'sklearn': SklearnGbmRegressorWrapper,
    'ours_vanilla': OurFastGbmRegressorWrapper,
    'ours_kfold': OurFastKfoldGbmRegressorWrapper})

GBM_CLASSIFIERS = Models(True, {
    'lgbm': LgbmGbmClassifierWrapper,
    'xgboost': XgboostGbmClassifierWrapper,
    'catboost': CatboostGbmClassifierWrapper,
    'sklearn': SklearnGbmClassifierWrapper,
    'ours_vanilla': OurFastGbmClassifierWrapper,
    'ours_kfold': OurFastKfoldGbmClassifierWrapper})

RF_REGRESSORS = Models(False, {
    'sklearn': SklearnRfRegressorWrapper,
    'ours_vanilla': OurFastRfRegressorWrapper,
    'ours_kfold': OurFastKfoldRfRegressorWrapper})

RF_CLASSIFIERS = Models(False, {
    'sklearn': SklearnRfClassifierWrapper,
    'ours_vanilla': OurFastRfClassifierWrapper,
    'ours_kfold': OurFastKfoldRfClassifierWrapper})
