"""
Functions to split a dataset without mixing classes and maintaining distributions
"""

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from tqdm import tqdm


def get_all_combs(file_sz, test_ratio, left_files=None, eps=0.02):
    if left_files is None:
        left_files = file_sz.keys().tolist()
    full_sz = file_sz.sum()
    curr_sz = 0
    curr_files = []
    valid_tests = get_valid_tests(
        curr_files, left_files, curr_sz, full_sz, file_sz, test_ratio, eps=eps
    )
    return valid_tests


def get_valid_tests(curr_files, left_files, curr_sz, full_sz, file_sz, test_ratio, eps=0.02):
    """Get all combinations that achieve a test ratio in [test_ratio, test_ratio + eps]"""
    valid_tests = []
    for fid, file in enumerate(left_files):
        sz = file_sz.loc[file] + curr_sz
        if (sz / full_sz) >= test_ratio:
            if (sz / full_sz) > (test_ratio + eps):
                continue
            valid_tests.append((set(curr_files + [file]), sz))
        else:
            valid_tests.extend(
                get_valid_tests(
                    curr_files + [file],
                    left_files[(fid + 1) :],
                    sz,
                    full_sz,
                    file_sz,
                    test_ratio,
                )
            )
    return valid_tests


def eval_splits(df: pd.DataFrame, valid_splits, files_id: str, labels: list) -> pd.DataFrame:
    """
    1. Find combination of files (or groups) that reach `test_ratio` of the data
    2. Find which of these combinations has the most similar topic distribution
    Returns a DataFrame with "MSE" and "MAPE" columns, so the user can choose manually

    df: DataFrame to split
    valid_splits: list with valid_splits obtained from `get_valid_tests`
    files_id: ID of files
    labels: list of labels which we want to preserve the distribution of
    """
    cat_dists = []
    for comb in tqdm(valid_splits):
        test_data = df[df[files_id].isin(comb)]
        test_dist = test_data[labels].sum() * 100 / len(test_data)
        cat_dists.append(test_dist.to_frame().T)

    res = pd.concat(cat_dists, keys=range(len(cat_dists)), ignore_index=True)
    res.loc["objective"] = df[labels].sum() * 100 / len(df)

    res["MSE"] = res.apply(lambda row: mean_squared_error(row, res.loc["objective"]), axis=1)
    res["MAPE"] = res.apply(
        lambda row: mean_absolute_percentage_error(row, res.loc["objective"]), axis=1
    )
    return res
