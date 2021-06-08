from time import time

from numpy.ma import arange
from numpy.random import seed
from sklearn.model_selection import train_test_split, KFold

from experiments.default_config import VAL_RATIO
from experiments.experiment_runner import run_experiment
from experiments.utils import make_dirs


def experiment_configurator(config, interaction_configuration = False):
    models = get_variants(config.predictors.is_gbm, config.one_hot)
    start = time()
    make_dirs([config.results_dir])
    for model_name, model_variants in models.items():
        print(f'Working on experiment : {model_name}')
        exp_dir = config.results_dir / model_name
        make_dirs([exp_dir])
        for variant in model_variants:
            config._set_attributes(model_name=model_name, variant=variant)
            seed(config.seed)
            if interaction_configuration:
                configurations = get_interaction_configuration(config, model_name, variant, exp_dir)
            else:
                configurations = get_configurations(config, model_name, variant, exp_dir)
            experiments_counter = 0
            for configuration in configurations:
                experiments_counter += 1
                seed(experiments_counter)
                if config.exp_results_path.exists():
                    continue
                run_experiment(configuration)

    end = time()
    print(f"run took {end - start} seconds")


def get_configurations(config, model_name, variant, exp_dir):
    if config.multiple_experimens:
        return multiple_experiments_configuration(config, model_name, variant, exp_dir)
    elif config.kfold_flag:
        return get_kfold_configuration(config, model_name, variant, exp_dir)
    else:
        return get_regular_configuration(config, model_name, variant, exp_dir)


def get_variants(is_gbm: bool, one_hot: bool):
    variants = ['mean_imputing', 'one_hot'] if one_hot else ['mean_imputing']
    if is_gbm:
        return dict(catboost=['vanilla'], lgbm=['vanilla'], xgboost=variants, sklearn=variants,
                    ours_vanilla=['_'], ours_kfold=['_'])
    return dict(sklearn=variants, ours_vanilla=['_'], ours_kfold=['_'])


def multiple_experiments_configuration(config, model_name, variant, exp_dir):
    for i in range(config.n_experiments):
        X, y = config.get_x_y()
        config._set_attributes(
            exp_results_path=exp_dir / F"{model_name}_{variant}_{i}.yaml",
            data=train_test_split(X, y, test_size=VAL_RATIO))
        yield config


def get_kfold_configuration(config, model_name, variant, exp_dir):
    X, y = config.get_x_y()
    kf = KFold(n_splits=config.kfolds, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        config._set_attributes(
            exp_results_path=exp_dir / F"{model_name}_{variant}_{i}.yaml",
            data=(X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]))
        yield config


def get_regular_configuration(config, model_name, variant, exp_dir):
    X, y = config.get_x_y()
    config._set_attributes(
        exp_results_path=exp_dir / F"{model_name}_{variant}.yaml",
        data=train_test_split(X, y, test_size=VAL_RATIO))
    yield config


def get_interaction_configuration(config, model_name, variant, exp_dir):
    for a in range(11):
    # for a in arange(8,10.5,0.5):5
        for i in range(config.n_experiments):
            X, y = config.get_x_y(a)
            config._set_attributes(
                exp_results_path=exp_dir / F"{model_name}_{variant}_{a}_{i}.yaml",
                data=train_test_split(X, y, test_size=VAL_RATIO))
            yield config
