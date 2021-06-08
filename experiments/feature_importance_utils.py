import shap
import xgboost as xgb
from catboost import Pool
from numpy import mean, square, array, abs
from numpy.random import permutation
from pandas import Series, DataFrame

from experiments.default_config import N_PERMUTATIONS


def compute_mse(model_name, model, X, y, transform_x=True, categorical_features=None):
    if transform_x:
        temp_x = X.copy()
        if model_name == 'xgboost':
            temp_x = xgb.DMatrix(temp_x)
        elif model_name == 'catboost':
            if categorical_features:
                temp_x = Pool(temp_x, cat_features=categorical_features)
            else:
                temp_x = Pool(temp_x)
        return mean(square(y - model.predict(temp_x)))
    return mean(square(y - model.predict(X)))


def permutation_feature_importance(model_name, model, X, y, categorical_features=None):
    results = {}
    mse = compute_mse(model_name, model, X, y, categorical_features=categorical_features)
    for col in X.columns:
        permutated_x = X.copy()
        random_feature_mse = []
        for i in range(N_PERMUTATIONS):
            permutated_x[col] = permutation(permutated_x[col])
            if model_name == 'xgboost':
                temp_x = xgb.DMatrix(permutated_x)
            elif model_name == 'catboost':
                temp_x = Pool(permutated_x, cat_features=categorical_features) if categorical_features else Pool(
                    permutated_x)
            else:
                temp_x = permutated_x
            random_feature_mse.append(
                compute_mse(model_name, model, temp_x, y, transform_x=False, categorical_features=categorical_features))
        results[col] = mean(array(random_feature_mse)) - mse
    results = Series(results)
    return results / results.sum()


def get_fi_gain(model_name, reg, X_train):
    if model_name == 'ours':
        fi_gain = Series(reg.compute_feature_importance(method='gain'))
    elif model_name == 'sklearn':
        fi_gain = Series(reg.feature_importances_, index=X_train.columns)
    elif model_name == 'xgboost':
        fi_gain = Series(reg.get_score(importance_type='gain'))
    else:  # model_name == 'catboost'
        fi_gain = Series(reg.feature_importances_, index=reg.feature_names_)
    if fi_gain.sum() != 0:
        fi_gain /= fi_gain.sum()
    return fi_gain


def get_fi_permutation(model_name, reg, X_train, y_train, X_test, y_test, categorical_features):
    if model_name == 'catboost':
        fi_permutation_train = permutation_feature_importance(model_name, reg, X_train, y_train,
                                                              categorical_features=categorical_features)
        fi_permutation_test = permutation_feature_importance(model_name, reg, X_test, y_test,
                                                             categorical_features=categorical_features)
    else:
        fi_permutation_train = permutation_feature_importance(model_name, reg, X_train, y_train)
        fi_permutation_test = permutation_feature_importance(model_name, reg, X_test, y_test)
    return fi_permutation_train, fi_permutation_test


def get_shap_values(model, x, columns=None):
    if columns is None:
        columns = x.columns
    abs_shap_values = DataFrame(shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(x),
                                columns=columns).apply(abs)
    return abs_shap_values.mean() / abs_shap_values.mean().sum()


def get_fi_shap(model_name, reg, X_train, X_test, y_train, y_test, cat_features):
    if model_name in ['sklearn', 'xgboost']:
        fi_shap_train = get_shap_values(reg, X_train)
        fi_shap_test = get_shap_values(reg, X_test)
    elif model_name == 'catboost':
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_test, y_test, cat_features=cat_features)
        fi_shap_train = get_shap_values(reg, train_pool, columns=X_train.columns)
        fi_shap_test = get_shap_values(reg, val_pool, columns=X_test.columns)
    else:
        fi_shap_train, fi_shap_test = None, None
    return fi_shap_train, fi_shap_test
