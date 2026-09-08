from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

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

