import yaml

from algorithms import LEARNING_RATE
from algorithms.Tree.fast_tree.bining import BinMapper
from algorithms.Tree.utils import get_num_cols
from experiments.default_config import MAX_DEPTH, N_ESTIMATORS, LEARNING_RATE
from experiments.utils import transform_categorical_features


def run_experiment(config):
    X_train, X_test, y_train, y_test, original_dtypes = get_data(config)
    model = get_model(config, original_dtypes)
    model.fit(X_train, y_train)
    model.compute_fi_gain()
    permutation_train, permutation_test = get_permutation_train_test(config.compute_permutation,
                                                                     model, X_train, y_train, X_test, y_test,
                                                                     original_dtypes)
    if config.save_results:
        results_dict = dict(ntrees=int(model.get_n_trees()),
                            nleaves=int(model.get_n_leaves()),
                            error=float(model.compute_error(X_test, y_test)),
                            gain=model.compute_fi_gain().to_dict(),
                            permutation_train=permutation_train,
                            permutation_test=permutation_test,
                            shap_train=model.compute_fi_shap(X_train, y_train).to_dict(),
                            shap_test=model.compute_fi_shap(X_test, y_test).to_dict())
        with open(config.exp_results_path, 'w') as file:
            documents = yaml.dump(results_dict, file)


def get_data(config):
    X_train, X_test, y_train, y_test = config.data
    preprocessing_pipeline = config.preprocessing_pipeline(0.5, config.columns_to_remove)
    preprocessing_pipeline.fit(X_train)
    X_train, X_test = map(preprocessing_pipeline.transform, [X_train, X_test])
    original_dtypes = X_train.dtypes

    if config.model_name.startswith('ours'):
        X_train, x_test = bin_numeric_features(X_train, X_test, config.contains_num_features)

    X_train, X_test = transform_categorical_features(X_train, X_test, y_train, config.variant)
    return X_train, X_test, y_train, y_test, original_dtypes


def get_model(config, dtypes):
    if config.predictors.is_gbm:
        return config.predictors.models_dict[config.model_name] \
            (config.variant, dtypes, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS,
             learning_rate=LEARNING_RATE, subsample=1.)
    return config.predictors.models_dict[config.model_name](config.variant, dtypes, N_ESTIMATORS)


def bin_numeric_features(x_train, x_test, contains_num_features):
    if contains_num_features:
        print("binning data")
        num_cols = get_num_cols(x_train.dtypes)
        bin_mapper = BinMapper(max_bins=256, random_state=42)
        x_train.loc[:, num_cols] = bin_mapper.fit_transform(x_train.loc[:, num_cols].values)
        x_test.loc[:, num_cols] = bin_mapper.transform(x_test.loc[:, num_cols].values)
        return x_train, x_test
    return x_train, x_test


def get_permutation_train_test(compute_permutation, model, X_train, y_train, X_test, y_test, original_dtypes):
    permutation_train = {col: 0 for col in original_dtypes.index}
    permutation_test = {col: 0 for col in original_dtypes.index}
    if compute_permutation:
        permutation_train = model.compute_fi_permutation(X_train, y_train).to_dict()
        permutation_test = model.compute_fi_permutation(X_test, y_test).to_dict()
    return permutation_train, permutation_test
