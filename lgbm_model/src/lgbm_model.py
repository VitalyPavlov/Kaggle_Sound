import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import label_ranking_average_precision_score
from lightgbm import LGBMClassifier
import json


SEED = 42
N_FOLD = 5
NUM_CLASSES = 80
PATH = "/Users/vivpavlov/Documents/PythonScripts/Kaggle/Sound"


def label_encoding(train_meta):
    y_encoded = np.zeros((len(train_meta), NUM_CLASSES)).astype(int)
    for i, label in enumerate(train_meta):
        y_encoded[i, label] = 1
    return y_encoded


def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0,
        scores[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return 'lwlrap', overall_lwlrap, True


def lgbm_model(X_train, y_train, y_encoded):
    """ LightGBM model """
    clfs = []
    folds = StratifiedKFold(n_splits=N_FOLD,
                            shuffle=True,
                            random_state=SEED)

    with open('../data/lgbm_params.json', 'r') as f:
        best_params = json.load(f)

    oof_preds = np.zeros((len(X_train), NUM_CLASSES))
    for fold_, (trn_, val_) in enumerate(folds.split(y_train, y_train)):
        trn_x, trn_y, trn_y_enc = X_train[trn_], y_train.iloc[trn_], y_encoded[trn_]
        val_x, val_y, val_y_enc = X_train[val_], y_train.iloc[val_], y_encoded[val_]

        clf = LGBMClassifier(**best_params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            verbose=1,
            eval_metric='multi_logloss',
            early_stopping_rounds=50
        )
        clfs.append(clf)

        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        _, loss, _ = calculate_overall_lwlrap_sklearn(val_y_enc, oof_preds[val_])
        print('no {}-fold loss: {}'.format(fold_ + 1, loss))

    _, loss, _ = calculate_overall_lwlrap_sklearn(y_encoded, oof_preds)
    print('MCC: {:.5f}'.format(loss))

    return clfs, loss, oof_preds


def fit():
    X_train = np.load('../data/train_data.npy')

    train_meta = pd.read_csv('../data/train_meta.csv')
    y_train = train_meta['target']
    y_encoded = label_encoding(y_train)

    clfs, loss, y_pred = lgbm_model(X_train, y_train, y_encoded)
    return clfs, loss, y_pred


def predict(clfs):
    X_test = np.load('../data/test_data.npy')
    test = pd.read_csv(PATH + '/sample_submission.csv')

    preds = None
    for clf in clfs:
        if preds is None:
            preds = clf.predict_proba(X_test, num_iteration=clf.best_iteration_)
        else:
            preds += clf.predict_proba(X_test, num_iteration=clf.best_iteration_)

    preds = preds / len(clfs)

    test.iloc[:, 1:] = preds
    test.head()
    return test


def main():
    clfs, _, _ = fit()
    y_pred = predict(clfs)
    y_pred.to_csv('../data/submission.csv', index=False)


if __name__ == "__main__":
    main()
