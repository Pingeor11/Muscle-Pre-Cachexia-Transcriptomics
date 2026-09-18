from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
import pandas as pd

def run_loocv_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 500,
    random_state: int = 51,
):
    """Leave-one-out cross validated Random Forest.

    Fits one RandomForestClassifier per held-out sample, so feature importance is
    averaged across folds rather than read off a single fit on all the data. Returns
    (mean_importance sorted descending, loocv_accuracy, per-fold detail DataFrame).
    """
    loo = LeaveOneOut()
    importance_sum = pd.Series(0.0, index=X.columns)
    fold_records = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        clf.fit(X_train, y_train)

        prediction = clf.predict(X_test)[0]
        is_correct = prediction == y_test.iloc[0]

        importance_sum = importance_sum.add(pd.Series(clf.feature_importances_, index=X.columns))
        fold_records.append({
            "held_out_sample": X_test.index[0],
            "true_label": y_test.iloc[0],
            "predicted_label": prediction,
            "correct": is_correct,
        })

    n_folds = len(X)
    mean_importance = (importance_sum / n_folds).sort_values(ascending=False)
    fold_detail = pd.DataFrame(fold_records)
    accuracy = fold_detail["correct"].mean()

    return mean_importance, accuracy, fold_detail
