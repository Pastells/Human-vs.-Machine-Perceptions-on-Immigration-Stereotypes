import os

import matplotlib.pyplot as plt
import seaborn as sns

from utils import config


def init_plot_params(font_scale=1.8, **kwargs):
    config.FIG_PARAMS.update(**kwargs)
    plt.rcParams.update(config.FIG_PARAMS)
    sns.set_theme(rc=config.FIG_PARAMS)
    sns.set_style("white")
    sns.set_context("paper", font_scale=font_scale)
    sns.set_palette(config.PALETTE)


def savefig(name, suffix=None):
    sns.despine()
    if suffix:
        name = name + "_" + suffix
    plt.tight_layout()
    plt.savefig(
        os.path.join(config.BASE_DIR, "results/figures", name + ".png"),
        bbox_inches="tight",
    )
    plt.clf()
