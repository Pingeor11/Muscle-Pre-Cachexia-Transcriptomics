from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.tree import DecisionTreeClassifier
from skrebate import ReliefF
import pandas as pd
import numpy as np

def weight_by_information_gain(X: pd.DataFrame,y: pd.Series):
    scores = mutual_info_classif(X, y, discrete_features = False, n_neighbors = 3, random_state = 51)

    return pd.Series(data = scores, index = X.columns)

def weight_by_chi_squared(X: pd.DataFrame,y: pd.Series):
    chi_2_stats = chi2(X, y)[0]

    return pd.Series(data = chi_2_stats, index = X.columns)

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
    overall_majority = 0
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


def weight_by_relief(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    relief = ReliefF(n_neighbors = 3, categorical_threshold = 0)
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
