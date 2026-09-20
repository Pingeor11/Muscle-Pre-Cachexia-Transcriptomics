from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import pandas as pd
import numpy as np

def _solver_kwargs(l1_ratio: float) -> dict:
    """liblinear for the pure endpoints (fast, exact), saga for any true Elastic Net mix."""
    if l1_ratio in (0.0, 1.0):
        return {"solver": "liblinear"}
    return {"solver": "saga", "max_iter": 5000}

def cpm_log_normalize(X: pd.DataFrame) -> pd.DataFrame:
    """Library-size-normalize (counts per million) then log2(CPM + 1).

    Needed whenever comparing raw counts across two independently-sequenced
    datasets — sequencing depth differs between studies even when both are
    nominally "raw counts", and a linear model's scale-sensitive penalty
    shouldn't be picking up that technical difference as if it were biology.
    Standard RNA-seq practice, not specific to any one model here.
    """
    library_sizes = X.sum(axis=1)
    cpm = X.div(library_sizes, axis=0) * 1e6
    return np.log2(cpm + 1)

def fit_final_model(X: pd.DataFrame, y: pd.Series, C: float, l1_ratio: float = 1.0, random_state: int = 51):
    """Fit one model on all of X/y (no held-out fold) — for scoring genuinely new data.

    `run_loocv_lasso` deliberately never does this (LOOCV refits per fold, by design,
    so there's no single persisted model) — use this instead when you actually want to
    apply a trained model to an external dataset rather than cross-validate on your own.
    Returns (fitted_scaler, fitted_classifier).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(l1_ratio=l1_ratio, C=C, random_state=random_state, **_solver_kwargs(l1_ratio))
    clf.fit(X_scaled, y)

    return scaler, clf

def run_loocv_lasso(X: pd.DataFrame, y: pd.Series, C: float = 0.3, l1_ratio: float = 1.0, random_state: int = 51):
    """Leave-one-out cross validated L1/Elastic Net logistic regression.

    `l1_ratio=1.0` (default) is pure LASSO — its L1 penalty zeroes out most gene
    coefficients. `l1_ratio` strictly between 0 and 1 is Elastic Net, which tends to
    keep or drop correlated genes together instead of arbitrarily picking one and
    zeroing the rest (LASSO's behavior) — relevant here since real gene pairs among
    the top attribute-weighting hits turned out to be strongly correlated (e.g.
    Dbp-Ky r=0.95). Either way this reports how often each gene survives with a
    nonzero coefficient across the 8 leave-one-out folds (stability selection),
    alongside its mean coefficient and LOOCV accuracy. Features are standardized
    within each fold (fit on train only) since penalty strength is scale-sensitive.

    `solver` is picked automatically: `liblinear` for the pure endpoints (fast, exact),
    `saga` for any true Elastic Net mix (only solver that supports it, needs more
    iterations to converge).

    `C` is not a safe one-size-fits-all default, and the right value is different for
    LASSO vs Elastic Net at the same `l1_ratio` — on synthetic LASSO data there is a
    sharp cliff where the model selects 0 genes (collapses to predicting the majority
    class) just below C=0.3, and jumps to ~80 genes just above it. Use `sweep_C` first
    to find a sane value for whatever `l1_ratio`/data you're running.
    """
    loo = LeaveOneOut()
    selected_counts = pd.Series(0.0, index=X.columns)
    coef_sum = pd.Series(0.0, index=X.columns)
    fold_records = []
    solver_kwargs = _solver_kwargs(l1_ratio)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(l1_ratio=l1_ratio, C=C, random_state=random_state, **solver_kwargs)
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

def sweep_C(X: pd.DataFrame, y: pd.Series, C_values, l1_ratio: float = 1.0, random_state: int = 51) -> pd.DataFrame:
    """Try a range of C values and report how many genes get selected at each.

    Run this before picking a C for `run_loocv_lasso` — the right value is data
    dependent and the transition from 0 selected genes to too many can be abrupt.
    Re-run this separately for each `l1_ratio` you use (LASSO vs Elastic Net) — the
    same C means something different at a different l1_ratio.
    """
    rows = []
    for C in C_values:
        _, accuracy, fold_detail = run_loocv_lasso(X, y, C=C, l1_ratio=l1_ratio, random_state=random_state)
        rows.append({
            "C": C,
            "mean_genes_per_fold": fold_detail["n_genes_selected"].mean(),
            "min_genes_per_fold": fold_detail["n_genes_selected"].min(),
            "max_genes_per_fold": fold_detail["n_genes_selected"].max(),
            "accuracy": accuracy,
        })
    return pd.DataFrame(rows)

def hedges_g(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Small-sample-corrected standardized effect size (Hedges' g) per gene.

    Independent of any model or penalty — computed directly from expression values
    split by label, so the same result can be joined onto both the LASSO and Elastic
    Net summary tables for a direct, model-agnostic comparison. Positive = higher in
    the y==1 group (C26), negative = higher in y==0 (SHAM).
    """
    group1 = X[y == 1]
    group0 = X[y == 0]
    n1, n0 = len(group1), len(group0)

    mean_diff = group1.mean() - group0.mean()
    pooled_sd = np.sqrt(((n1 - 1) * group1.var(ddof=1) + (n0 - 1) * group0.var(ddof=1)) / (n1 + n0 - 2))
    cohens_d = mean_diff / pooled_sd

    small_sample_correction = 1 - (3 / (4 * (n1 + n0) - 9))
    return cohens_d * small_sample_correction

def permutation_test_accuracy(X: pd.DataFrame, y: pd.Series, C: float, l1_ratio: float = 1.0, random_state: int = 51):
    """Exact permutation-null test for LOOCV accuracy.

    At n=8 with a balanced 4/4 label split, there are only C(8,4)=70 possible
    balanced label assignments — few enough to enumerate all of them exactly,
    rather than randomly sampling a subset (an approximate test). Reruns the full
    LOOCV pipeline once per permutation, using the same C/l1_ratio as the real
    analysis (never re-tuned per permutation — that would be circular).

    Returns (observed_accuracy, null_accuracies, p_value), where p_value is the
    fraction of all 70 balanced label assignments — including the true one — that
    achieve accuracy at least as high as the true labels' accuracy.
    """
    n = len(y)
    n_pos = int(y.sum())
    true_label_tuple = tuple(y.values)

    null_accuracies = []
    observed_accuracy = None

    for pos_combo in combinations(range(n), n_pos):
        permuted = np.zeros(n, dtype=int)
        for i in pos_combo:
            permuted[i] = 1
        permuted_y = pd.Series(permuted, index=y.index)

        _, accuracy, _ = run_loocv_lasso(X, permuted_y, C=C, l1_ratio=l1_ratio, random_state=random_state)
        null_accuracies.append(accuracy)

        if tuple(permuted_y.values) == true_label_tuple:
            observed_accuracy = accuracy

    null_accuracies = np.array(null_accuracies)
    p_value = (null_accuracies >= observed_accuracy).mean()

    return observed_accuracy, null_accuracies, p_value
