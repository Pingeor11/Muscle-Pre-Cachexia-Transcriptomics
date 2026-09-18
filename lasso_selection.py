from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import pandas as pd

def run_loocv_lasso(X: pd.DataFrame, y: pd.Series, C: float = 0.3, random_state: int = 51):
    """Leave-one-out cross validated L1 (LASSO) logistic regression.

    LASSO's L1 penalty zeroes out most gene coefficients, so this reports how often
    each gene survives with a nonzero coefficient across the 8 leave-one-out folds
    (stability selection), alongside its mean coefficient and LOOCV accuracy. Features
    are standardized within each fold (fit on train only) since L1 penalty strength is
    scale-sensitive.

    `C` is not a safe one-size-fits-all default — on the synthetic data there is a
    sharp cliff where the model selects 0 genes (collapses to predicting the majority
    class) just below C=0.3, and jumps to ~80 genes just above it. Use `sweep_C` first
    to find a sane value for whatever data you're running on.
    """
    loo = LeaveOneOut()
    selected_counts = pd.Series(0.0, index=X.columns)
    coef_sum = pd.Series(0.0, index=X.columns)
    fold_records = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C, random_state=random_state)
        clf.fit(X_train_scaled, y_train)

        prediction = clf.predict(X_test_scaled)[0]
        is_correct = prediction == y_test.iloc[0]

        coefs = pd.Series(clf.coef_[0], index=X.columns)
        selected_counts = selected_counts.add((coefs != 0).astype(float))
        coef_sum = coef_sum.add(coefs)

        fold_records.append({
            "held_out_sample": X_test.index[0],
            "true_label": y_test.iloc[0],
            "predicted_label": prediction,
            "correct": is_correct,
            "n_genes_selected": int((coefs != 0).sum()),
        })

    n_folds = len(X)
    summary = pd.DataFrame({
        "SelectionFreq": selected_counts / n_folds,
        "MeanCoef": coef_sum / n_folds,
    })
    summary["AbsMeanCoef"] = summary["MeanCoef"].abs()
    summary = summary.sort_values(["SelectionFreq", "AbsMeanCoef"], ascending=[False, False])

    fold_detail = pd.DataFrame(fold_records)
    accuracy = fold_detail["correct"].mean()

    return summary, accuracy, fold_detail

def sweep_C(X: pd.DataFrame, y: pd.Series, C_values, random_state: int = 51) -> pd.DataFrame:
    """Try a range of C values and report how many genes get selected at each.

    Run this before picking a C for `run_loocv_lasso` — the right value is data
    dependent and the transition from 0 selected genes to too many can be abrupt.
    """
    rows = []
    for C in C_values:
        _, accuracy, fold_detail = run_loocv_lasso(X, y, C=C, random_state=random_state)
        rows.append({
            "C": C,
            "mean_genes_per_fold": fold_detail["n_genes_selected"].mean(),
            "min_genes_per_fold": fold_detail["n_genes_selected"].min(),
            "max_genes_per_fold": fold_detail["n_genes_selected"].max(),
            "accuracy": accuracy,
        })
    return pd.DataFrame(rows)
