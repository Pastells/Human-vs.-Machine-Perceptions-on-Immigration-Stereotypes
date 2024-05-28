import glob
import os
import warnings
from argparse import Namespace
from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from utils import config
from utils.results import DfResults, compute_metrics1, compute_metrics2

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# MODELS
# ============================================================

BASELINES = ["all_ones", "all_zeros", "random", "tfidf", "fast"]
BERTS = [
    "beto",
    "robertuito",
    "roberta_bne",
    "bertin",
    "mbert",
    "roberta_large_bne",
]
ZERO_SHOT = ["gpt_3.5", "gpt_4", "llama", "mDeBERTa"]

# ============================================================
# TEST FILES
# ============================================================

st_test = pd.read_csv(
    os.path.join(config.BASE_DIR, "data/stereohoax/test_split_context_soft.csv"),
    index_col=0,
)
st_unaggregated = pd.read_csv(
    os.path.join(config.BASE_DIR, "data/stereohoax", "stereohoax_unaggregated.csv"),
    index_col=0,
)
ST_TEST = st_test.join(st_unaggregated[["implicit_soft", "contextual_soft", "url_soft"]]).fillna(0)

# ============================================================


def add_gold(df, conf, task, contextual=False):
    if task == 1:
        res = compute_metrics1(
            df,
            conf.target,
            soft_labels=True,
            contextual=contextual,
            target_pred=conf.target,
        )
    elif task == 2:
        res = compute_metrics2(
            df,
            conf.target,
            [conf.target, "implicit"],
            contextual=contextual,
            labels_pred=[conf.target, "implicit"],
        )
    else:
        res = compute_metrics2(
            df,
            conf.target,
            [conf.target] + conf.topics,
            contextual=contextual,
            labels_pred=[conf.target] + conf.topics,
        )
    return res


def compute_metrics_wrapper(
    df: pd.DataFrame,
    conf: Namespace,
    task: int,
    soft_labels: bool = False,
    contextual: bool = False,
    implicit_categories: bool = False,
    bootstrap_f1: bool = False,
) -> dict:
    """
    Wrapper for compute_metrics1 (task 1) and compute_metrics2 (tasks 2 and 3)
    """
    if task == 1:
        res = compute_metrics1(
            df,
            conf.target,
            soft_labels=soft_labels,
            contextual=contextual,
            implicit_categories=implicit_categories,
            bootstrap_f1=bootstrap_f1,
        )
    elif task == 2:
        res = compute_metrics2(df, conf.target, [conf.target, "implicit"], contextual=contextual)
    else:
        res = compute_metrics2(df, conf.target, [conf.target] + conf.topics, contextual=contextual)
    return res


def join_wrapper(file: str, data: str) -> pd.DataFrame:
    if data == "stereohoax":
        return join_by_index(file, ST_TEST)
    else:
        raise ValueError("Must be stereohoax")


def join_by_index(file: str, test: pd.DataFrame) -> pd.DataFrame:
    """Join results file with rest of test columns by index"""
    df = pd.read_csv(file, index_col=0)
    columns = [c for c in df.columns if c in test.columns]
    df = df.join(test.drop(columns=columns), validate="1:1")
    return df


def glob_files(data, model, task, context=False, baselines=True, results_filename="") -> List[str]:
    """
    Returns list of files for a given data, model and task
    """
    if results_filename:
        results_filename = os.path.join("results/metrics", results_filename)
        if os.path.isfile(results_filename):
            ref_time = os.stat(results_filename).st_mtime
        else:
            ref_time = 0
    else:
        ref_time = 0

    folders = [
        "fine_tuning",
        "fine_tuning_256",
        "fine_tuning_512",
        "zero_shot",
    ]
    if baselines:
        folders += ["baselines"]
    if context:
        folders += [
            "fine_tuning_context",
            "fine_tuning_context_append",
            "fine_tuning_context_no_fill",
            "fine_tuning_context_top200",
            "fine_tuning_context_no_same_fill",
            "fine_tuning_context_no_same_no_fill",
        ]

    files = []
    for folder in folders:
        files += sorted(glob.glob(f"results/{data}/{folder}/*{model}*task{task}.csv"))

    files = [f for f in files if os.stat(f).st_mtime > ref_time]
    return files


def create_metrics(
    data: str,
    task: int,
    name: str,
    context=True,
    implicit_categories=False,
    bootstrap_f1=False,
    recompute_all=False,
):
    """Loop over all the results of the given task and compute the metrics"""
    conf = config.get_conf(data)
    contextual = context if data == "stereohoax" else False
    results = DfResults(st_test, conf, task=task, contextual=contextual)
    contexts = conf.contexts_results + ["soft", "debug"]
    models = BASELINES + [bert + "-" + cont for bert in BERTS for cont in contexts] + ZERO_SHOT
    df = None
    for model in tqdm(models):
        for file in glob_files(
            data,
            model,
            task,
            context,
            results_filename="" if recompute_all else name,
        ):
            try:
                df = join_wrapper(file, data)
                file = os.path.basename(file)[:-10]
                soft_labels = "soft" in file
                res = compute_metrics_wrapper(
                    df,
                    conf,
                    task,
                    soft_labels=soft_labels,
                    contextual=contextual,
                    implicit_categories=implicit_categories,
                    bootstrap_f1=bootstrap_f1,
                )
                results.append(file, res)
            except pd.errors.MergeError:
                print(f"MergeError in {file}")
            except KeyError as e:
                print(f"KeyError in {file}")
                raise e

    if df is not None and recompute_all:
        results.append("gold", add_gold(df, conf, task, contextual=contextual))
    if "ICM" in results.df.columns:
        results.df.insert(
            2,
            "ICM_norm",
            (results.df["ICM"] - results.df["ICM"].min())
            / (results.df["ICM"].max() - results.df["ICM"].min()),
        )

    if recompute_all:
        results.save(name, append=False)
    else:
        results.save(name, append=True)


# -----------------------------------------------------
# STYLE
# -----------------------------------------------------


def color_bad(val, color="red", thres=0.5):
    """
    Takes a scalar and returns a string with
    the css property `'color: color'` for
    values < thres
    strings, black otherwise.
    """
    if isinstance(val, float) and val < thres:
        col = color
    else:
        col = "black"
    return "color: %s" % col


def highlight_max(s, color="green"):
    """highlight the maximum in a Series"""
    is_max = s == s.max()
    return [f"background-color: {color}" if v else "" for v in is_max]


def better_than_no_cont(s, n_contexts, n_baselines=6, n_models=4, color="lime"):
    ss = []
    for i in range(n_models):
        s_i = s[n_baselines + i * n_contexts : n_baselines + (i + 1) * n_contexts].reset_index(
            drop=True
        )
        ss.extend((s_i > s_i[0]).tolist())

    ss = [False] * n_baselines + ss
    return [f"background-color: {color}" if v else "" for v in ss]


def best_in_model(s, n_contexts, n_baselines=6, n_models=4, color="lime"):
    ss = []
    for i in range(n_models):
        s_i = s[n_baselines + i * n_contexts : n_baselines + (i + 1) * n_contexts].reset_index(
            drop=True
        )
        ss.extend((s_i == s_i.max()).tolist())

    ss = [False] * n_baselines + ss
    return [f"background-color: {color}" if v else "" for v in ss]


def highlight_str(s, string, column):
    has_str = pd.Series(data=False, index=s.index)
    has_str[column] = string in s.loc[column]
    return ["background-color: lightgrey" if has_str.any() else "" for _ in has_str]


def model_column_color(s):
    conditions = [
        s.str.contains("Beto"),
        s.str.contains("Roberta"),
        s.str.contains("Robertuito"),
    ]
    choices = ["orange", "violet", "chocolate", "lightcoral"]
    colors = np.select(conditions, choices, default="turquoise")
    return [f"background-color: {color}" for color in colors]


def model_column_txt(column, data, remove=None):
    column = (
        column.str.replace(f"{data}-", "")
        .str.title()
        .str.replace("Svc", "SVC")
        .str.replace("Tfidf", "TFIDF")
        .str.replace("_", " ")
        .str.replace("-", " - ")
        .str.replace("+", " + ")
        .str.replace(" gram", "-gram")
        .str.replace("knn ", "k-nn, k=")
        .str.replace("Lr", "LR ")
        .str.replace("E - ", "e-")
        .str.replace("S42", "")
        .str.replace("Ep20", "")
    )
    for r in remove or []:
        column = column.str.replace(r, "")
    return column


def metrics_style_context_task1(data, remove=None):
    import dataframe_image as dfi

    conf = config.get_conf(data)
    df = pd.read_csv(
        os.path.join(config.BASE_DIR, "results/metrics", f"metrics_{data}-context-task1.csv")
    )
    df["model"] = model_column_txt(df["model"], data, remove=remove)
    s = (
        df.rename(columns={c: c.title().replace("_", " ") for c in df.columns})
        .style.applymap(color_bad)
        .apply(better_than_no_cont, n_contexts=len(conf.contexts) + 1)
        .apply(highlight_str, string="No Context", column="Model", axis=1)
        .set_properties(**{"text-align": "left"})
        .set_table_styles([dict(selector="th", props=[("text-align", "left")])])
        .apply(highlight_max)
        .apply(model_column_color, axis=0, subset="Model")
        .format(precision=3)
        .hide(axis="index")
    )
    try:
        dfi.export(
            s,
            os.path.join(config.BASE_DIR, "results/metrics", f"metrics_{data}-context-task1.png"),
            dpi=config.DPI,
        )
    except KeyError:
        print(f"KeyError creating metrics figure with context for {data}")
    return s


def metrics_style_task1(data, remove=None):
    import dataframe_image as dfi

    conf = config.get_conf(data)
    df = pd.read_csv(os.path.join(config.BASE_DIR, "results/metrics", f"metrics_{data}-task1.csv"))
    df["model"] = model_column_txt(df["model"], data, remove=remove)
    s = (
        df.rename(columns={c: c.title().replace("_", " ") for c in df.columns})
        .style.applymap(color_bad)
        .apply(best_in_model, n_contexts=len(conf.contexts) + 1)
        .set_properties(**{"text-align": "left"})
        .set_table_styles([dict(selector="th", props=[("text-align", "left")])])
        .apply(highlight_max)
        .apply(model_column_color, axis=0, subset="Model")
        .format(precision=3)
        .hide(axis="index")
    )
    dfi.export(
        s,
        os.path.join(config.BASE_DIR, "results/metrics", f"metrics_{data}-task1.png"),
        dpi=config.DPI,
    )
    return s


def main():
    bootstrap_f1 = False
    recompute_all = True

    data_task_imp = (("stereohoax", 1, False),)

    for data, task, implicit_categories in data_task_imp:
        print(data, task)
        if not recompute_all:
            print("Computing only changed files")
        create_metrics(
            data,
            task,
            f"metrics_stereohoax-task{task}.csv",
            context=True,
            implicit_categories=implicit_categories,
            bootstrap_f1=bootstrap_f1,
            recompute_all=recompute_all,
        )


if __name__ == "__main__":
    main()
