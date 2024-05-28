"""
- DfResults class to store results and compute metrics
- Metrics for task 1
- Metrics for task 2
- Metrics for task 3
- Metrics for fine tuning
"""

import os
import warnings
from argparse import Namespace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import entropy
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)

from utils import config

warnings.filterwarnings("ignore", category=FutureWarning)


class DfResults:
    """Dataframe with results

    test: pd.DataFrame test set
    conf: configuration NameSpace for given corpus
    task: 1 - Predict stereotypes (yes/no)
          2 - Predict stereotypes implicitness
          3 - Predict stereotypes topic (hierarchical multi-label classification)
    """

    def __init__(
        self,
        test: pd.DataFrame,
        conf: Namespace,
        task: int = 1,
        contextual: bool = False,
    ):
        self.test = test.copy()
        self.conf = conf
        self.task = task
        self.contextual = contextual
        self.df = pd.DataFrame()

    def append(self, name: str, results: dict):
        """Appends row to self.df, with name of the model and results"""
        self.df = pd.concat(
            [
                self.df,
                pd.DataFrame(
                    [[name] + list(results.values())],
                    columns=["model"] + list(results.keys()),
                ),
            ],
            ignore_index=True,  # otherwise all have index=0
        ).fillna(0)

    def add(
        self,
        name: str,
        pred=None,
        pred_soft=None,
        softmax_confidence=None,
        seed_confidence=None,
        temperature=1.0,
        save=True,
        folder=None,
    ):
        """Adds model predictions to test DataFrame
        aggregates by implicitness and computes metrics
        task 1:
            pred: can be either array (len(test)) or number (all ones, all zeros)

        task 2:
            pred: can be either array (len(test)) or number (all ones, all zeros)

        task 3:
            pred: must be an (len(test), len(conf.topics)) array.
                  otherwise, individual columns can be added to self.test
                  for each topic

        pred_soft: soft labels predictions

        Possible confidence scores, for any task:
        - softmax_confidence: from softmax
        - seed_confidence: from different seeds
        """
        target_pred = self.conf.target + "_pred"
        save_columns = self.conf.indexes + [self.conf.target, target_pred]

        if self.task == 1:
            if pred is None:
                raise ValueError("add must be given a pred for task 1")

            self.test[target_pred] = pred

            if pred_soft is not None:
                target_pred_soft = self.conf.target + "_pred_soft"
                target_soft = self.conf.target + "_soft"
                self.test[target_pred_soft] = pred_soft
                save_columns.extend([target_soft, target_pred_soft])

            results = compute_metrics1(
                self.test,
                self.conf.target,
                contextual=self.contextual,
            )

        elif self.task == 2:
            imp = "implicit"
            imp_pred = "implicit_pred"
            save_columns.extend([imp, imp_pred])

            if pred:
                self.test[imp_pred] = pred

            if pred_soft is not None:
                imp_soft = "implicit_soft"
                imp_pred_soft = "implicit_pred_soft"
                self.test[imp_pred_soft] = pred_soft
                save_columns.extend([imp_soft, imp_pred_soft])

            results = compute_metrics2(
                self.test,
                self.conf.target,
                [self.conf.target, imp],
                contextual=self.contextual,
            )
        else:
            topics_pred = [topic + "_pred" for topic in self.conf.topics]
            all_topics = list(zip(self.conf.topics, topics_pred))
            all_topics = [item for sublist in all_topics for item in sublist]
            save_columns.extend(all_topics)

            if pred:
                self.test[topics_pred] = pred

            results = compute_metrics2(
                self.test,
                self.conf.target,
                [self.conf.target] + self.conf.topics,
                contextual=self.contextual,
            )

        def _add_confidence(conf_name, confidence, save_columns):
            self.test[f"{conf_name}_confidence"] = confidence
            save_columns.append(f"{conf_name}_confidence")
            results[f"{conf_name}_confidence_mean"] = confidence.mean()
            results[f"{conf_name}_confidence_std"] = confidence.std()
            return save_columns

        if seed_confidence is not None:
            save_columns = _add_confidence("seed", seed_confidence, save_columns)

        if softmax_confidence is not None:
            save_columns = _add_confidence("softmax", softmax_confidence, save_columns)
            self.test["temperature"] = None
            self.test.iloc[0, self.test.columns.get_loc("temperature")] = temperature
            save_columns.append("temperature")

        if save:
            if folder is None:
                folder = ""
            self.test[save_columns].to_csv(
                os.path.join(
                    config.BASE_DIR,
                    "results",
                    self.conf.data,
                    folder,
                    f"{name}-task{self.task}.csv",
                ),
                index=False,
            )
        self.append(name, results)

    def save(self, file: str, append=True):
        """Saves results to file
        append: if True, append to file if already exists"""
        file = os.path.join("results/metrics", file)
        exists = os.path.isfile(file)
        if append and exists:
            self.df.to_csv(file, mode="a", header=False, float_format="%.3f", index=False)
        else:
            self.df.to_csv(file, float_format="%.3f", index=False)


# -------------------------------------------------------------------------------------


def aggregate(df, target, labels, labels_pred, column, metrics_fun, suffixes):
    """Aggregates df by `column`, assuming target == 1,
    and compute metrics for both groups.
    Returns [metric1_neg, metric1_pos, metric2_neg, metric2_pos, ...]

    target: True/False target used for task1
    labels: target (task1) or list of topics (task2)
    labels_pred: labels + "_pred"
    column: field by which to aggregate
    metrics_fun: function to compute the metrics for the aggregated df
    suffixes: suffixes for the metrics, e.g., ("explicit", "implicit")
    """
    negative = df[(df[target] == 1) & (df[column] < 0.5)]
    positive = df[(df[target] == 1) & (df[column] >= 0.5)]

    res_neg = metrics_fun(negative, labels, labels_pred)
    res_pos = metrics_fun(positive, labels, labels_pred)

    res_neg = {k + suffixes[0]: v for k, v in res_neg.items()}
    res_pos = {k + suffixes[1]: v for k, v in res_pos.items()}

    results = {**res_neg, **res_pos}
    return results


def cross_entropy(targets, predictions, epsilon=1e-12) -> float:
    """
    Cross entropy of predictions (q) with respect to targets (p)
    H(p,q) = -sum(p log(q))
    """
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce


def jensen_shannon_divergence(target, predictions):
    js = (entropy(target, predictions) + entropy(predictions, target)) / 2
    return js


# -------------------------------------------------------------------------------------
# Task 1 metrics
# -------------------------------------------------------------------------------------


def bootstrap(
    df: pd.DataFrame,
    target: str,
    target_pred: str,
    n_loops: int = 1000,
    sample_size: float = 1.0,
    soft_labels: bool = False,
) -> dict:
    """
    Simple unparametric bootstrap to get 95% confidence interval for f1 metrics
    """

    def bootstrap_iteration(df, sample_size, target, target_pred, soft_labels):
        _df = df.sample(frac=sample_size, replace=True)
        res = _compute_metrics1_util(_df, target, target_pred, soft_labels, only_f1=True)
        return res

    results_list = Parallel(n_jobs=-1)(
        delayed(bootstrap_iteration)(df, sample_size, target, target_pred, soft_labels)
        for _ in range(n_loops)
    )

    results_df = pd.DataFrame(results_list)

    final_results = {}
    for column in results_df.columns:
        final_results[column] = results_df[column].mean().round(3)
        conf_interval = np.round(np.percentile(results_df[column].to_numpy(), [2.5, 97.5]), 3)
        final_results[f"{column}_min"] = conf_interval[0]
        final_results[f"{column}_max"] = conf_interval[1]

    return final_results


def _compute_metrics1_util(
    df: pd.DataFrame,
    target: str,
    target_pred: str,
    soft_labels: bool = False,
    contextual: bool = False,
    implicit_categories: bool = False,
    only_f1: bool = False,
) -> dict:
    """
    contextual: aggregate for contextual
    implicit_categories: aggregate for implicit categories
    only_f1: compute only f1 metrics
    """
    results = metrics1(df, target, target_pred, soft_labels, only_f1=only_f1)

    # Aggregate by implicitness
    implicit_explicit = aggregate(
        df,
        target,
        target,
        target_pred,
        "implicit",
        metrics1_aggregated,
        suffixes=("_explicit", "_implicit"),
    )

    results = {**results, **implicit_explicit}

    if implicit_categories:
        for imp_cat in config.common["implicit_categories"]:
            name = "_implicit_" + imp_cat[:5]
            res_implicit_context = aggregate(
                df,
                target,
                target,
                target_pred,
                imp_cat,
                metrics1_aggregated,
                suffixes=("_implicit_", name),
            )
            res_implicit_context = (
                {"f1" + name: res_implicit_context["f1" + name]}
                if "f1" + name in res_implicit_context
                else {}
            )
            results = {**results, **res_implicit_context}

    # Aggregate by contextual
    if contextual:
        res_contextual = aggregate(
            df,
            target,
            target,
            target_pred,
            "contextual",
            metrics1_aggregated,
            suffixes=("_no_contextual", "_contextual"),
        )
        results = {**results, **res_contextual}

    return results


def compute_metrics1(
    df: pd.DataFrame,
    target: str = "stereo",
    soft_labels: bool = False,
    contextual: bool = False,
    implicit_categories: bool = False,
    target_pred: str = "",
    bootstrap_f1=False,
) -> dict:
    """
    contextual: aggregate for contextual
    bootstrap_f1: if true return avg, min and max from bootstrap for f1 metrics
    """
    if not target_pred:
        target_pred = target + "_pred"

    results = _compute_metrics1_util(
        df=df,
        target=target,
        target_pred=target_pred,
        soft_labels=soft_labels,
        contextual=contextual,
        implicit_categories=implicit_categories,
    )

    if not bootstrap_f1:
        return results

    res_bootstrap = bootstrap(df, target, target_pred, soft_labels=soft_labels)
    return {**results, **res_bootstrap}


def analyze_confidence(df, target, conf_name="softmax", iaa=False):
    confidence = f"{conf_name}_confidence"
    if iaa:
        df["iaa"] = np.where((0.2 < df[target + "_soft"]) & (df[target + "_soft"] < 0.8), 0.67, 1)
    res = {}
    res[confidence] = df[confidence].mean()
    res[confidence + "_std"] = df[confidence].std()

    names = ("TN", "FP", "FN", "TP")
    names = [f"{n}_{confidence}" for n in names]
    for i in range(2):
        for j in range(2):
            data = df[(df[target] == i) & (df[target + "_pred"] == j)]
            agg = data[confidence].agg(["mean", "std"])
            res[f"{names[i*2+j]}"] = agg[0]
            res[f"{names[i*2+j]}_std"] = agg[1]
            if iaa:
                agr = data[data["iaa"] == 1][confidence].agg(["mean", "std"])
                dis = data[data["iaa"] < 1][confidence].agg(["mean", "std"])
                res[f"{names[i*2+j]}_agree_mean"] = agr[0]
                res[f"{names[i*2+j]}_agree_std"] = agr[1]
                res[f"{names[i*2+j]}_disagree_mean"] = dis[0]
                res[f"{names[i*2+j]}_disagree_std"] = dis[1]

    return res


def metrics1_soft(df, target, target_pred):
    target_soft = target + "_soft"
    target_pred_soft = target_pred + "_soft"

    labels = df[target_soft].to_numpy()
    pred = df[target_pred_soft].to_numpy()

    res = {}
    res["cross_entropy"] = cross_entropy(labels, pred)
    res["MSE"] = mean_squared_error(labels, pred, squared=False)
    names = ("TN", "FP", "FN", "TP")
    for i in range(2):
        for j in range(2):
            data = df[(df[target] == i) & (df[target + "_pred"] == j)]
            labels = data[target_soft].to_numpy()
            pred = data[target_pred_soft].to_numpy()
            res[names[i * 2 + j] + "_cross_entropy"] = cross_entropy(labels, pred)
            res[names[i * 2 + j] + "_MSE"] = mean_squared_error(labels, pred, squared=False)
    return res


def metrics1(df, target, target_pred, soft_labels=False, only_f1=False, c_metrics=False) -> dict:
    """
    Metrics for task1
    - F-1: negative and positive class
    - precision: negative and positive class
    - recall: negative and positive class
    - accuracy
    - Roc AUC
    - Matthews Correlation Coefficient

    - Cross entropy (soft_labels)
    - MSE (soft_labels)

    - cPrecision, cRecall and cF1 if c_metrics is True

    only_f1: compute only f1 metrics: f1_pos, f1_neg, f1_explicit, f1_implicit
    """
    labels = df[target].to_numpy()
    pred = df[target_pred].to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average=None, zero_division=0
    )
    results = {
        "f1_neg": f1[0],
        "f1_pos": f1[1],
    }
    if not only_f1:
        res = {
            "roc_auc": roc_auc_score(labels, pred),
            "mcc": matthews_corrcoef(labels, pred),
            "precision_neg": precision[0],
            "precision_pos": precision[1],
            "recall_neg": recall[0],
            "recall_pos": recall[1],
            "accuracy": accuracy_score(labels, pred),
        }
        results = {**results, **res}

    for conf in ("softmax_confidence", "seed_confidence"):
        if conf in df.columns:
            res = analyze_confidence(df, target, conf.split("_")[0])
            results = {**results, **res}

    if soft_labels:
        res = metrics1_soft(df, target, target_pred)
        results = {**results, **res}

    elif "confidence" in df.columns and c_metrics:
        c_precision, c_recall, c_f1 = c_precision_recall_fscore(df, target=target)
        results["c_precision_neg"] = c_precision[0]
        results["c_precision_pos"] = c_precision[1]
        results["c_recall_neg"] = c_recall[0]
        results["c_recall_pos"] = c_recall[1]
        results["c_f1_neg"] = c_f1[0]
        results["c_f1_pos"] = c_f1[1]

    return results


def metrics1_aggregated(df, target, target_pred) -> dict:
    """Metrics for task1 aggregated by implicitness or context"""
    labels = df[target].to_numpy()
    pred = df[target_pred].to_numpy()
    results = {"accuracy": accuracy_score(labels, pred)}
    return results


def c_precision_recall_fscore(df, beta=1, target="stereo"):
    """
    Implementation of:
        Probabilistic Extension of Precision, Recall, and F1 Score
        for More Thorough Evaluation of Classification Models
        by Yacouby R and Axman D
        https://aclanthology.org/2020.eval4nlp-1.9/
    """

    def _get_M(data, target="stereo"):
        """Confidence scores (probabilities)"""
        target_pred = target + "_pred"
        df = data.copy()
        df["prob"] = (df["confidence"] + 1) / 2
        df["prob_pos"] = np.where(df[target_pred] == 1, df["prob"], 1 - df["prob"])
        df["prob_neg"] = 1 - df["prob_pos"]
        M = df[["prob_neg", "prob_pos"]].to_numpy()
        return M

    def _get_pCM(df, M, label, target="stereo"):
        """Confidence Confusion Matrix"""
        target_pred = target + "_pred"
        cTP = M[(df[target_pred] == label) & (df[target] == label)][:, label].sum()
        cFP = M[(df[target_pred] == label) & (df[target] != label)][:, label].sum()
        return cTP, cFP

    M = _get_M(df, target)
    beta2 = beta**2

    precision = []
    recall = []
    fscore = []
    for label in range(2):
        cTP, cFP = _get_pCM(df, M, label, target)
        prec = cTP / (cTP + cFP)
        rec = cTP / df[target].value_counts()[label]
        fb = (1 + beta2) * (prec * rec) / ((beta2 * prec) + rec)

        precision.append(prec)
        recall.append(rec)
        fscore.append(fb)

    return precision, recall, fscore


# -------------------------------------------------------------------------------------
# Tasks 2 and 3 metrics
# -------------------------------------------------------------------------------------


def compute_metrics2(df, target, labels, contextual=False, labels_pred=None) -> dict:
    if not labels_pred:
        labels_pred = [label + "_pred" for label in labels]
    results = metrics2(df, labels, labels_pred)

    # Aggregate by implicitness
    implicit_explicit = aggregate(
        df,
        target,
        labels,
        labels_pred,
        "implicit",
        metrics2,
        suffixes=("_explicit", "_implicit"),
    )
    results = {**results, **implicit_explicit}

    # Aggregate by contextual
    if contextual:
        res_contextual = aggregate(
            df,
            target,
            labels,
            labels_pred,
            "contextual",
            metrics2,
            suffixes=("_no_contextual", "_contextual"),
        )
        results = {**results, **res_contextual}

    return results


def metrics2(df, labels, labels_pred) -> dict:
    """Metrics for task2 and task3"""
    labels = df[labels].to_numpy()
    pred = df[labels_pred].to_numpy()
    results = {
        "ICM": icm(labels, pred),
        "Hierarchical-F": hierarchical_f(labels, pred),
        "Propensity-F": propensity_f(labels, pred),
    }
    return results


def propensity_f(labels, pred, A=0.55, B=1.5):
    """
    This Propensity F measure assumes all classes are binary and encoded with 0s and 1s
    Note 1: True and False are considered separate classes
    Note 2: Root is added as a smoothing term
    Note 3: this is not a hierarchical metric
    """
    # Add empty set (all ones for smoothing) and false class (1st column == 0)
    empty_set = np.ones((labels.shape[0], 1))
    labels = np.hstack((empty_set, np.abs(labels[:, :1] - 1), labels))
    pred = np.hstack((empty_set, np.abs(pred[:, :1] - 1), pred))

    Nc = labels.sum(axis=0, keepdims=True)  # Only ones are considered
    gs_c_ones = labels.astype(bool)
    pred_c_ones = pred.astype(bool)
    hit_mask = np.logical_and(gs_c_ones, pred_c_ones)

    C = (np.log2(labels.shape[0]) - 1) * ((B + 1) ** A)
    pc = 1 / (1 + C * np.exp(-A * np.log2(Nc + B)))

    inv_pc = 1 / pc
    inv_pc = np.broadcast_to(inv_pc, labels.shape)
    numerator = np.multiply(inv_pc, hit_mask).sum(1)
    prop_p = numerator / np.multiply(inv_pc, pred_c_ones).sum(1)
    prop_r = numerator / np.multiply(inv_pc, gs_c_ones).sum(1)
    prop_f = 2 * prop_p * prop_r / (prop_p + prop_r)

    return prop_f.mean()


def hierarchical_f(labels, pred):
    # Add empty set (all ones for smoothing) and false class (1st column == 0)
    empty_set = np.ones((labels.shape[0], 1))
    labels = np.hstack((empty_set, np.abs(labels[:, :1] - 1), labels))
    pred = np.hstack((empty_set, np.abs(pred[:, :1] - 1), pred))

    hit_mask = np.logical_and(labels.astype(bool), pred.astype(bool))
    n_hits = hit_mask.sum(1)

    p_f = n_hits / pred.sum(1)
    r_f = n_hits / labels.sum(1)
    f_f = 2 * p_f * r_f / (p_f + r_f)

    return f_f.mean()


def icm(labels, pred, alpha1=2, alpha2=2, beta=3):
    def ic(sets, ic_nodes):
        ic_sets = np.multiply(sets.astype(bool), ic_nodes)
        # In our scenario only True has descendants (the categories),
        # so we need to sum the IC of all categories
        ic_descendents = ic_sets[:, 3:].sum(1)
        # The IC of the ascendant (True) of each pair needs to be subtracted |C| - 1 times
        n_categories = sets[:, 3:].sum(1)
        total_ic_sets = ic_descendents - (n_categories - 1) * ic_sets[:, 2]
        # If there are no categories, we will have only have False in the set. Need to add IC(False)
        total_ic_sets += ic_sets[:, 1]
        # Remember that the IC(ROOT) = 0, and the lso(False, any_category) = ROOT so there is no need to subtract it
        return total_ic_sets

    # Add empty set (all ones for smoothing) and false class (1st column == 0)
    empty_set = np.ones((labels.shape[0], 1))
    labels = np.hstack((empty_set, np.abs(labels[:, :1] - 1), labels))
    pred = np.hstack((empty_set, np.abs(pred[:, :1] - 1), pred))

    # [ROOT, FALSE, TRUE, C1, C2, ..., C10]
    ic_c = np.broadcast_to(-np.log2(labels.sum(0) / labels.shape[0]), labels.shape)

    ic_y = ic(labels, ic_c)
    ic_y_pred = ic(pred, ic_c)
    ic_y_union_y_pred = ic(np.logical_or(labels, pred), ic_c)

    icm = alpha1 * ic_y + alpha2 * ic_y_pred - beta * ic_y_union_y_pred
    return icm.mean()


# -------------------------------------------------------------------------------------
# Fine-tuning metrics
# -------------------------------------------------------------------------------------


def compute_metrics_soft(output):
    pred, labels = output
    ce = cross_entropy(labels, pred)
    rmse = mean_squared_error(labels, pred, squared=False)
    return {"ce": ce, "rmse": rmse}


def compute_metrics_hard(output):
    pred, labels = output
    pred = np.argmax(pred, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average=None, zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision_neg": precision[0],
        "precision_pos": precision[1],
        "recall_neg": recall[0],
        "recall_pos": recall[1],
        "f1_neg": f1[0],
        "f1_pos": f1[1],
        "f1": (f1[0] + f1[1]) / 2,
        "mcc": matthews_corrcoef(labels, pred),
    }
