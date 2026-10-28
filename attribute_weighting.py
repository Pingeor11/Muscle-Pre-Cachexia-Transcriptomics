from sklearn.tree import DecisionTreeClassifier
from scipy.stats import chi2_contingency
from skrebate import ReliefF
import pandas as pd
import numpy as np

def default_n_neighbors(y: pd.Series, cap: int = 3) -> int:
    """Neighbor count for Relief, capped at `cap` but never >= the smaller class size.

    The 3 used at n=8 (4 SHAM/4 C26) breaks down on smaller resamples (e.g. stability
    selection's leave-one-pair-out 3v3 folds), where n_neighbors=3 exceeds what a class
    of 3 can supply once ReliefF excludes the point itself.
    """
    min_class_size = y.value_counts().min()
    return max(1, min(cap, min_class_size - 1))

def weight_by_chi_squared(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Chi-squared statistic on the same MDL-discretized bins used by IG/IGRatio/SU.

    Chi-squared needs a bins-vs-label contingency table, not raw continuous (possibly
    negative) expression values — sklearn's `chi2()` assumes non-negative count-like
    features, which doesn't hold here and doesn't match how RapidMiner's Weight by
    Chi Squared Statistic operator works (it discretizes numeric attributes first).
    """
    scores = {}
    for gene in X.columns:
        bins = mdl_discretize(X[gene], y)
        contingency = pd.crosstab(bins, y)
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            scores[gene] = 0.0
            continue
        chi2_stat, _, _, _ = chi2_contingency(contingency, correction=False)
        scores[gene] = chi2_stat
    return pd.Series(scores)

def weight_by_gini_index(X,y):
    scores = {}
    for gene in X.columns:
        tree = DecisionTreeClassifier(max_depth = 1, random_state = 51)
        tree.fit(X[[gene]], y)
        scores[gene] = tree.tree_.compute_feature_importances(normalize=False)[0]
    return(pd.Series(scores))

def majority_or_default(labels: pd.Series, default):
    counts = labels.value_counts()
    if len(counts) < 2:
        return counts.index[0]
    if counts.iloc[0] == counts.iloc[1]:
        return default
    return counts.index[0]


def weight_by_rule(X: pd.DataFrame, y: pd.Series):
    scores = {}
    overall_majority = majority_or_default(y, sorted(y.unique())[0])
    for gene in X.columns:
        values = X[gene]
        sorted_vals = pd.Series(values.unique()).sort_values()
        thresholds = [(a+b)/2 for a,b in zip(sorted_vals[:-1], sorted_vals[1:])]

        best_accuracy = 0.0
        for t in thresholds:
            left = (values <= t)
            right = ~left
            left_pred = majority_or_default(y[left], overall_majority)
            right_pred = majority_or_default(y[right], overall_majority)

            predictions = left.map({True: left_pred, False: right_pred})

            accuracy = (predictions == y).sum()/len(y)

            best_accuracy = max(best_accuracy, accuracy)

        scores[gene] = best_accuracy

    return pd.Series(scores)

def entropy(labels: pd.Series) -> float:
    counts = labels.value_counts()
    probs = counts/len(labels)
    return -sum(probs * np.log2(probs))

def mdl_candidate_boundaries(values: pd.Series, y: pd.Series) -> list[float]:
    paired = pd.DataFrame({"value": values, "label": y}).sort_values("value")
    vals = paired["value"].tolist()
    labels = paired["label"].tolist()

    boundaries = []
    for i in range(len(vals) - 1):
        if (vals[i] != vals[i+1]) and (labels[i] != labels[i+1]):
            boundaries.append((vals[i] + vals[i+1])/2)

    return boundaries

def weighted_entropy_after_split(values:pd.Series, y: pd.Series, boundary: float) -> float:
    left = (values <= boundary)
    right = ~left
    n = len(values)
    n_left = left.sum()
    n_right = n - n_left
    left_entropy = entropy(y[left])
    right_entropy = entropy(y[right])

    return (n_left/n) * left_entropy + (n_right/n) * right_entropy

def find_best_split(values: pd.Series, y: pd.Series):
    boundaries = mdl_candidate_boundaries(values, y)
    if not boundaries:
        return None, 0.0

    parent_entropy = entropy(y)

    best_boundary = None
    best_weighted_entropy = float("inf")
    for b in boundaries:
        weight = weighted_entropy_after_split(values, y , b)
        if (weight < best_weighted_entropy):
            best_weighted_entropy = weight
            best_boundary = b
    info_gain = parent_entropy - best_weighted_entropy
    return best_boundary, info_gain

def mdl_accept_split(values: pd.Series, y: pd.Series, boundary:float, info_gain: float) -> bool:
    left = (values <= boundary)
    right = ~left
    N = len(y)

    parent_entropy = entropy(y)
    left_entropy = entropy(y[left])
    right_entropy = entropy(y[right])

    k = y.nunique()
    k1 = y[left].nunique()
    k2 = y[right].nunique()

    delta = np.log2(3**k -2) - (k * parent_entropy - k1 * left_entropy - k2 * right_entropy)
    threshold = np.log2(N-1)/N + delta/N

    return info_gain > threshold

def mdl_boundaries_recursive(values: pd.Series, y: pd.Series) -> list[float]:
    boundary, info_gain = find_best_split(values, y)
    if boundary is None:
        return []
    if not mdl_accept_split(values, y, boundary, info_gain):
        return []

    left_mask = (values <= boundary)
    right_mask = ~left_mask

    left_boundaries = mdl_boundaries_recursive(values[left_mask], y[left_mask])
    right_boundaries = mdl_boundaries_recursive(values[right_mask], y[right_mask])

    return left_boundaries + [boundary] + right_boundaries

def mdl_discretize(values: pd.Series, y: pd.Series) -> pd.Series:
    boundaries = sorted(mdl_boundaries_recursive(values, y))

    bin_edges = [-np.inf] + boundaries + [np.inf]

    return pd.cut(values, bins = bin_edges, labels = False)

def weighted_entropy_across_bins(y: pd.Series, bins: pd.Series) -> float:
    n = len(y)
    weighted = 0.0
    for b in bins.unique():
        mask = bins == b
        weight = mask.sum() / n
        weighted += weight * entropy(y[mask])
    return weighted

def weight_by_information_gain(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Entropy-based IG on the same MDL-discretized bins as IGRatio/SU below.

    Kept consistent with weight_by_information_gain_ratio/weight_by_symmetric_uncertainty
    deliberately: those are both this same discretized-entropy IG divided by an entropy
    term, matching RapidMiner's Weight by Information Gain family. A continuous kNN-based
    estimator (e.g. sklearn's mutual_info_classif) is a different algorithm and noisier
    at n=8, so it isn't used here.
    """
    scores = {}
    h_label = entropy(y)
    for gene in X.columns:
        bins = mdl_discretize(X[gene], y)
        scores[gene] = h_label - weighted_entropy_across_bins(y, bins)
    return pd.Series(scores)

def weight_by_information_gain_ratio(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    scores = {}
    for gene in X.columns:
        bins = mdl_discretize(X[gene], y)
        h_feature = entropy(bins)

        if h_feature == 0:
            scores[gene] = 0.0
            continue

        ig = entropy(y) - weighted_entropy_across_bins(y, bins)
        scores[gene] = ig / h_feature

    return pd.Series(scores)


def weight_by_symmetric_uncertainty(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    scores = {}
    h_label = entropy(y)
    for gene in X.columns:
        bins = mdl_discretize(X[gene], y)
        h_feature = entropy(bins)
        ig = entropy(y) - weighted_entropy_across_bins(y, bins)

        denom = h_feature + h_label
        if denom == 0:
            scores[gene] = 0.0
            continue

        scores[gene] = 2 * ig / denom

    return pd.Series(scores)


def weight_by_relief(X: pd.DataFrame, y: pd.Series, n_neighbors: int | None = None) -> pd.Series:
    if n_neighbors is None:
        n_neighbors = default_n_neighbors(y)
    relief = ReliefF(n_neighbors = n_neighbors, categorical_threshold = 0)
    relief.fit(X.values, y.values)
    scores = relief.feature_importances_
    return pd.Series(data=scores, index = X.columns)

def min_max_normalize(scores: pd.Series) -> pd.Series:
    min_val = scores.min()
    max_val = scores.max()
    if max_val == min_val:
        return pd.Series(0.0, index = scores.index)

    return (scores - min_val)/(max_val - min_val)

def run_attribute_weighting(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    raw_scores = {
        "IG": weight_by_information_gain(X, y),
        "Chi2": weight_by_chi_squared(X, y),
        "Gini": weight_by_gini_index(X, y),
        "Rule": weight_by_rule(X, y),
        "IGRatio": weight_by_information_gain_ratio(X, y),
        "Uncertainty": weight_by_symmetric_uncertainty(X, y),
        "Relief": weight_by_relief(X, y),
    }
    normalized = {name: min_max_normalize(scores) for name, scores in raw_scores.items()}

    result = pd.DataFrame(normalized)
    result["WeightSum"] = result.sum(axis = 1)
    result["Rank"] = result["WeightSum"].rank(ascending = False)
    result["Selected"] = result["WeightSum"] > 5

    return result.sort_values("WeightSum", ascending=False)

def leave_one_pair_out_splits(y: pd.Series) -> list[pd.Index]:
    """All folds that drop one sample from each of the two classes.

    For the 4 SHAM / 4 C26 design this exhaustively enumerates all 4*4=16 balanced
    3v3 subsets, rather than relying on random bootstrap resampling at n=8.
    """
    classes = y.unique()
    if len(classes) != 2:
        raise ValueError("leave_one_pair_out_splits expects a binary label")

    idx_by_class = {c: y.index[y == c] for c in classes}

    splits = []
    for drop_a in idx_by_class[classes[0]]:
        for drop_b in idx_by_class[classes[1]]:
            splits.append(y.index.difference([drop_a, drop_b]))
    return splits

def random_subsample_splits(
    y: pd.Series, n_repeats: int = 40, drop_fraction: float = 0.25, random_state: int = 51
) -> list[pd.Index]:
    """Repeated stratified random subsampling — each split drops ~drop_fraction of each class.

    Generalizes leave_one_pair_out_splits to sample sizes where exhaustively enumerating
    every drop-one-pair combination is too slow, and too small a perturbation to be a
    meaningful stress test (e.g. after pooling several datasets: dropping 1 pair out of
    8 samples is a 25% perturbation, dropping 1 pair out of 30 is under 10%). Class
    balance is preserved in every split, same principle as leave_one_pair_out_splits.
    """
    rng = np.random.default_rng(random_state)
    classes = y.unique()
    if len(classes) != 2:
        raise ValueError("random_subsample_splits expects a binary label")

    idx_by_class = {c: y.index[y == c].to_numpy() for c in classes}

    splits = []
    for _ in range(n_repeats):
        keep_idx = []
        for idx in idx_by_class.values():
            n_drop = max(1, round(len(idx) * drop_fraction))
            drop = rng.choice(idx, size=n_drop, replace=False)
            keep_idx.append(np.setdiff1d(idx, drop))
        splits.append(pd.Index(np.concatenate(keep_idx)))
    return splits

def run_stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[pd.Index] | None = None,
    top_k: tuple[int, ...] = (25, 50),
) -> pd.DataFrame:
    """Rerun the full weighting pipeline on each leave-one-pair-out fold.

    StabilityScore (fraction of folds where WeightSum > 5) turned out not to
    discriminate well on synthetic data — that cutoff is so lenient at n=8 that most
    genes clear it in most folds regardless of whether they carry real signal. The
    TopKFreq columns (fraction of folds where the gene's Rank fell in the top k) were
    the metric that actually separated the synthetic-injected signal genes from noise,
    so use those, not StabilityScore, to build a shortlist.
    """
    if splits is None:
        splits = leave_one_pair_out_splits(y)

    n_splits = len(splits)
    selected_counts = pd.Series(0.0, index=X.columns)
    weight_sum_totals = pd.Series(0.0, index=X.columns)
    rank_totals = pd.Series(0.0, index=X.columns)
    top_k_counts = {k: pd.Series(0.0, index=X.columns) for k in top_k}

    for keep_idx in splits:
        fold_result = run_attribute_weighting(X.loc[keep_idx], y.loc[keep_idx])
        selected_counts = selected_counts.add(fold_result["Selected"].astype(float), fill_value=0.0)
        weight_sum_totals = weight_sum_totals.add(fold_result["WeightSum"], fill_value=0.0)
        rank_totals = rank_totals.add(fold_result["Rank"], fill_value=0.0)
        for k in top_k:
            top_k_counts[k] = top_k_counts[k].add((fold_result["Rank"] <= k).astype(float), fill_value=0.0)

    stability = pd.DataFrame({
        "StabilityScore": selected_counts / n_splits,
        "MeanWeightSum": weight_sum_totals / n_splits,
        "MeanRank": rank_totals / n_splits,
        **{f"Top{k}Freq": top_k_counts[k] / n_splits for k in top_k},
    })
    sort_col = f"Top{top_k[0]}Freq"
    return stability.sort_values([sort_col, "MeanRank"], ascending=[False, True])
