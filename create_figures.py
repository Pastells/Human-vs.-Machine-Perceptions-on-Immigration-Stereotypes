"""
Create all figures

"""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from matplotlib import MatplotlibDeprecationWarning
from matplotlib.collections import PolyCollection

from create_metrics import join_wrapper
from utils import config
from utils.plots import init_plot_params, savefig

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

init_plot_params()


# ===============
# Loss
# ===============


def read_trainer_logs(logs_name):
    pickle_file = os.path.join(
        config.BASE_DIR, "results/trainer_logs", "trainer_logs_" + logs_name + ".pickle"
    )
    with open(pickle_file, "rb") as handle:
        trainer_logs = pickle.load(handle)
    return trainer_logs


def loss_plot(logs, names, fig_name, suffix="", ylim=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = sns.color_palette(config.PALETTE)[:10]
    iter(
        [
            "-",
            ":",
            (0, (1, 1)),
            "-.",
            "--",
            (0, (3, 1, 1, 1)),
        ]
    )

    for i, log in enumerate(logs):
        # Comment this line if logs don't have name and args info
        log = log[1]
        name = (
            names[i]
            .replace(suffix, "")
            .replace("beto", "BETO")
            .replace("roberta", "RoBERTa-BNE")
            .replace("robertuito", "RoBERTuito")
        )
        epoch = np.array([t.get("epoch") for t in log])
        loss = np.array([t.get("loss") for t in log])
        eval_loss = np.array([t.get("eval_loss") for t in log])

        eval_epoch = epoch[eval_loss != None]  # noqa
        epoch = epoch[loss != None]  # noqa
        loss = loss[loss != None].astype(float)  # noqa
        eval_loss = eval_loss[eval_loss != None].astype(float)  # noqa
        ax.plot(
            epoch,
            loss,
            label=f"{name} training",
            color=colors[i],
            linewidth=2,
        )
        ax.plot(
            eval_epoch,
            eval_loss,
            label=f"{name} validation",
            linestyle=(0, (1, 1)),
            color=colors[i],
            linewidth=2,
        )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if ylim:
        plt.ylim([0, ylim])
    plt.legend(
        fontsize=9,
        bbox_to_anchor=(0.5, 1.05),
        loc="upper center",
        ncol=2,
    )
    savefig(fig_name)


# ===============
# Confidence
# ===============


def normalize_soft_df(df, column):
    col = df[column]
    df[column[:-4] + "norm_soft"] = (col - col.min()) / (col.max() - col.min())
    df[column[:-4] + "clip_soft"] = (
        df[column[:-4] + "norm_soft"] * (0.95257413 - 0.04742587) + 0.04742587
    )
    return df


def annotate_pearsonr(data, x, y, **kws):
    r, p = sp.stats.pearsonr(data[x], data[y])
    ax = plt.gca()
    ax.text(0.3, 0.3, "r={:.2f}, p={:.2g}".format(r, p), transform=ax.transAxes)


def confidence_plot_hard(name, folder, data="stereohoax"):
    print(name)
    conf = config.get_conf(data)
    x_label = "iaa"
    y_label = "softmax_confidence"
    x_label_name = "IAA"
    y_label_name = "Confidence"

    df = join_wrapper(
        os.path.join(
            config.BASE_DIR,
            f"results/{data}",
            folder,
            name + ".csv",
        ),
        data,
    )
    df["implicit_str"] = np.where(df.implicit == 1, "Implicit Stereotype", "Explicit Stereotype")
    df["iaa"] = np.where(
        (0.2 < df[conf.target + "_soft"]) & (df[conf.target + "_soft"] < 0.8), 0.67, 1
    )

    df = df.rename(
        columns={
            x_label: x_label_name,
            y_label: y_label_name,
            conf.target + "_pred": "Prediction",
        }
    ).round(2)

    regplot_args = {
        "x": x_label_name,
        "y": y_label_name,
        "x_jitter": 0.03,
    }
    legend_args = {
        "fontsize": config.LEGEND_FONTSIZE,
        "loc": "upper center",
        "ncol": 2,
    }

    # 4 plots
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.1)
    colors = sns.color_palette(config.PALETTE)[:10]

    g = sns.regplot(
        data=df[(df[conf.target] == 0) & (df["Prediction"] == 0)],
        color=colors[0],
        marker="o",
        ax=axes[0, 0],
        **regplot_args,
    )
    g.set(xlabel=None)
    g.axes.set_title("Negative Prediction", fontsize=config.SUBTITLE_FONTSIZE)

    sns.regplot(
        data=df[(df[conf.target] == 1) & (df["Prediction"] == 0)],
        color=colors[1],
        line_kws={"ls": "--"},
        marker="^",
        ax=axes[0, 0],
        **regplot_args,
    ).set(xlabel=None)
    g = sns.regplot(
        data=df[(df[conf.target] == 0) & (df["Prediction"] == 1)],
        color=colors[0],
        marker="o",
        ax=axes[0, 1],
        **regplot_args,
    )
    g.set(xlabel=None, ylabel=None)
    g.axes.set_title("Positive Prediction", fontsize=config.SUBTITLE_FONTSIZE)
    sns.regplot(
        data=df[(df[conf.target] == 1) & (df["Prediction"] == 1)],
        color=colors[1],
        line_kws={"ls": "--"},
        marker="^",
        ax=axes[0, 1],
        **regplot_args,
    ).set(xlabel=None, ylabel=None)

    # Negative
    g = sns.regplot(
        data=df[(df[conf.target] == 1) & (df.implicit_soft < 0.1) & (df["Prediction"] == 0)],
        color=colors[2],
        marker="*",
        ax=axes[1, 0],
        **regplot_args,
    )
    g = sns.regplot(
        data=df[
            (df[conf.target] == 1)
            & (df.implicit_soft > 0.1)
            & (df.implicit_soft < 0.3)
            & (df["Prediction"] == 0)
        ],
        color=colors[3],
        line_kws={"ls": "--"},
        marker="s",
        ax=axes[1, 0],
        **regplot_args,
    )
    g = sns.regplot(
        data=df[(df[conf.target] == 1) & (df.implicit_soft > 0.3) & (df["Prediction"] == 0)],
        color=colors[4],
        line_kws={"ls": ":"},
        marker="D",
        ax=axes[1, 0],
        **regplot_args,
    )

    # Positive
    g = sns.regplot(
        data=df[(df[conf.target] == 1) & (df.implicit_soft < 0.1) & (df["Prediction"] == 1)],
        color=colors[2],
        marker="*",
        ax=axes[1, 1],
        **regplot_args,
    ).set(ylabel=None)
    g = sns.regplot(
        data=df[
            (df[conf.target] == 1)
            & (df.implicit_soft > 0.1)
            & (df.implicit_soft < 0.3)
            & (df["Prediction"] == 1)
        ],
        color=colors[3],
        line_kws={"ls": "--"},
        marker="s",
        ax=axes[1, 1],
        **regplot_args,
    ).set(ylabel=None)
    g = sns.regplot(
        data=df[(df[conf.target] == 1) & (df.implicit_soft > 0.3) & (df["Prediction"] == 1)],
        color=colors[4],
        line_kws={"ls": ":"},
        marker="D",
        ax=axes[1, 1],
        **regplot_args,
    ).set(ylabel=None)
    legend_args = {
        "fontsize": config.LEGEND_FONTSIZE,
        "ncol": 2,
        "columnspacing": 1,
    }
    fig.legend(
        labels=[
            "No Stereotype",
            "No Stereotype fit",
            "_no_legend_",
            "Stereotype",
            "Stereotype fit",
        ],
        bbox_to_anchor=(0.17, 0.555),
        loc="upper left",
        **legend_args,
    )

    fig.legend(
        labels=["_no_legend_"] * 12
        + [
            "Explicit",
            "Explicit fit",
            "_no_legend_",
            r"Implicit-one-vote",
            r"Implicit-one-vote fit",
            "_no_legend_",
            r"Implicit-by-majority",
            r"Implicit-by-majority fit",
        ],
        bbox_to_anchor=(0.9, 0.455),
        loc="lower right",
        **legend_args,
    )
    savefig(name, "iaa")


def confidence_plot_soft(name, folder, data="stereohoax"):
    print(name)
    conf = config.get_conf(data)
    x_label = conf.target + "_soft"
    y_label = conf.target + "_pred_norm_soft"
    x_label_name = "Annotator Soft Labels"
    y_label_name = "Predicted Soft Label"

    df = join_wrapper(
        os.path.join(
            config.BASE_DIR,
            f"results/{data}",
            folder,
            name + ".csv",
        ),
        data,
    )

    df["implicit_str"] = np.where(df.implicit == 1, "Implicit Stereotype", "Explicit Stereotype")
    df = normalize_soft_df(df, conf.target + "_pred_soft")
    df = df.rename(columns={x_label: x_label_name, y_label: y_label_name}).round(2)

    # 1) Violin
    sns.violinplot(data=df, x=x_label_name, y=y_label_name)
    savefig(name, "confidence_violin")

    # 3.2) Implicit Soft
    deep_blue = sns.color_palette("colorblind")[0]
    light_palette = sns.light_palette(deep_blue, n_colors=4)[::-1]

    _, ax = plt.subplots(1, 1, figsize=(5, 4))
    ihatch = iter(["", "...", r"\\"] * 2)
    g = sns.violinplot(
        data=df[df[conf.target] == 1],
        x=x_label_name,
        y=y_label_name,
        hue="implicit_soft",
        ax=ax,
        palette=light_palette,
    )
    _ = [i.set_hatch(next(ihatch)) for i in g.get_children() if isinstance(i, PolyCollection)]
    plt.legend(
        fontsize=config.LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, 1.05),
        loc="upper center",
        ncol=2,
        # columnspacing=1,
        labels=[
            "_no_legend_",
            "Explicit",
            "_no_legend_",
            "_no_legend_",
            "_no_legend_",
            "_no_legend_",
            r"Implicit-one-vote",
            "_no_legend_",
            "_no_legend_",
            "_no_legend_",
            "_no_legend_",
            r"Implicit-by-majority",
        ],
    )
    savefig(name, "confidence_violin_implicit_soft")


def confidence_plot_soft_gpt(name, name_gpt):
    x_label = "stereo_soft"
    y_label = "stereo_pred_norm_soft"
    x_label_name = "Annotator Soft Labels"
    y_label_name = "Prediction"
    COLUMNS = ["stereo", "stereo_soft", "stereo_pred_soft", "implicit", "implicit_soft"]

    df = join_wrapper(
        os.path.join(
            config.BASE_DIR,
            "results/stereohoax",
            "fine_tuning",
            name + ".csv",
        ),
        data="stereohoax",
    )[COLUMNS]
    df2 = join_wrapper(
        os.path.join(
            config.BASE_DIR,
            "results/stereohoax",
            "zero_shot",
            name_gpt + ".csv",
        ),
        data="stereohoax",
    )[COLUMNS]

    df["gpt"] = False
    df2["gpt"] = True
    df = pd.concat([df, df2]).round(2)

    df["implicit_str"] = np.where(df.implicit == 1, "Implicit Stereotype", "Explicit Stereotype")
    df = normalize_soft_df(df, "stereo_pred_soft")
    df = df.rename(columns={x_label: x_label_name, y_label: y_label_name}).round(2)

    # 1) Violin
    sns.violinplot(data=df, x=x_label_name, y=y_label_name, hue="gpt")
    plt.legend(
        fontsize=config.LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, 1.05),
        loc="upper center",
        ncol=2,
        labels=[
            "_no_legend_",
            "FT-SL",
            "_no_legend_",
            "_no_legend_",
            "_no_legend_",
            "_no_legend_",
            "GPT-4P",
        ],
    )
    savefig(name, "confidence_violin_gpt")

    # 3.2) Implicit Soft
    _y_label_name = "Predicted Probability"
    _df = df[df.gpt == True].rename(columns={y_label_name: _y_label_name}).round(2)
    orange = sns.color_palette("colorblind")[1]
    light_palette = sns.light_palette(orange, n_colors=4)[::-1]
    _, ax = plt.subplots(1, 1, figsize=(5, 4))
    ihatch = iter(["", "...", r"\\"] * 4)
    g = sns.violinplot(
        data=_df[_df.stereo == 1],
        x=x_label_name,
        y=_y_label_name,
        hue="implicit_soft",
        linewidth=0.7,
        cut=1.6,
        palette=light_palette,
        ax=ax,
    )
    _ = [i.set_hatch(next(ihatch)) for i in g.get_children() if isinstance(i, PolyCollection)]
    labels = [
        "Explicit",
        "Implicit-one-vote",
        "Implicit-by-majority",
    ]
    labels = ["_no_legend_"] + [
        labels[int(i / 5)] if i % 5 == 0 else "_no_legend_" for i in range(13)
    ]
    plt.legend(
        fontsize=config.LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, 1.05),
        loc="upper center",
        ncol=2,
        # columnspacing=1,
        labels=labels,
    )
    savefig(name, "confidence_violin_implicit_soft_gpt")



# ===============
# FIGURES
# ===============


def loss_plots():
    # Hard 1e-5
    logs_name = "st_hard_1e-5"
    fig_name = "hard_1e-5"
    names = (
        "beto, hard lr 1e-5",
        "roberta, hard lr 1e-5",
        "robertuito, hard lr 1e-5",
    )
    trainer_logs = read_trainer_logs(logs_name)
    loss_plot(trainer_logs, names, fig_name, suffix=", hard lr 1e-5", ylim=1)

    # Soft 2e-5
    logs_name = "st_soft_2e-5"
    fig_name = "soft_2e-5"
    names = (
        "beto, soft lr 2e-5",
        "roberta, soft lr 2e-5",
        "robertuito, soft lr 2e-5",
    )
    trainer_logs = read_trainer_logs(logs_name)
    loss_plot(trainer_logs, names, fig_name, suffix=", soft lr 2e-5")


def confidence_soft():
    for soft in ("stereohoax-roberta_bne-0_no_context_soft_s42_lr2e-05_ep20-task1",):
        confidence_plot_soft(soft, folder="fine_tuning")


def confidence_gpt():
    soft = "stereohoax-roberta_bne-0_no_context_soft_s42_lr2e-05_ep20-task1"
    gpt_prob = "gpt_4_prob-task1"
    confidence_plot_soft_gpt(soft, gpt_prob)


def confidence_hard():
    for hard, data in (
        (
            "stereohoax-roberta_bne-0_no_context_hard_s42_lr1e-05_ep20-task1",
            "stereohoax",
        ),
    ):
        confidence_plot_hard(hard, folder="fine_tuning", data=data)


if __name__ == "__main__":
    loss_plots()
    confidence_hard()
    confidence_soft()
    confidence_gpt()
