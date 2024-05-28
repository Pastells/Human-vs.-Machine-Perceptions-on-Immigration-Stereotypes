"""
Add context and soft-labels to given datasets
"""

import os

import numpy as np
import pandas as pd
from scipy.special import softmax

from utils import config


def add_soft(data, labels, annotator_suffix="_a"):
    _data = data.copy()
    if isinstance(labels, str):
        labels = (labels,)
    for label in labels:
        cols = [c for c in data.columns.to_list() if label + annotator_suffix in c]
        if not cols:
            continue
        df = _data[cols].copy()

        df["positive"] = df.sum(axis=1)
        df["negative"] = 3 - df["positive"]
        pos_neg = df[["positive", "negative"]].to_numpy()
        _data[label + "_soft"] = np.round(softmax(pos_neg, axis=1)[:, 0], 4)
    return _data


def fill_context(df: pd.DataFrame, text, contexts) -> pd.DataFrame:
    """
    - fill missing contexts with level above
    - remove contexts equal to text itself
    - concatenate all contexts into "concat_contexts"

    text: text column
    contexts: list of contexts"""
    _df = df.copy()
    _df["concat_contexts"] = _df[contexts[-1]]

    for i, cont in enumerate(contexts[:-1][::-1]):
        # remove context equal to text
        _df.loc[_df[text] == _df[cont], cont] = ""

        no_cont = _df[cont].isin(("", "0", "[]"))
        _df.loc[no_cont, cont] = ""
        print(
            f"missing {cont}:",
            len(_df[no_cont]),
            f"({len(_df[no_cont]) / len(_df) * 100:.0f}%)",
        )

        # concat context if it's not repated (same as context above)
        same_as_above = _df[cont] == _df[contexts[len(contexts) - 1 - i]]
        _df.loc[~same_as_above, "concat_contexts"] = (
            _df.loc[~same_as_above, cont] + " " + _df.loc[~same_as_above, "concat_contexts"]
        ).str.strip()

        # fill missing context
        _df.loc[no_cont, cont] = _df.loc[no_cont, contexts[len(contexts) - 1 - i]]
    return _df


# -----------------------------------------------------
# Main program
# -----------------------------------------------------


def main_stereohoax():
    print("DATA: stereohoax")
    conf = config.get_conf("stereohoax")
    for file in (
        "train_val_split.csv",
        "train_split.csv",
        "val_split.csv",
        "test_split.csv",
    ):
        print("FILE - ", file)
        file = os.path.join(conf.path, file)
        df = pd.read_csv(file)
        df = add_soft(df, (conf.target,))
        df = fill_context(df, "text", conf.contexts)
        df.to_csv(file[:-4] + "_context_soft.csv", index=False)


if __name__ == "__main__":
    main_stereohoax()
