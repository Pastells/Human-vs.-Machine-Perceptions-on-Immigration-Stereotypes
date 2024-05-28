import os
import shutil
from argparse import Namespace

SEED = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Figures
DPI = 600
PALETTE = "colorblind"
SUBTITLE_FONTSIZE = 18
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 10
FIG_PARAMS = {
    "text.usetex": shutil.which("latex") is not None,  # check if latex is installed
    "font.size": 10,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "figure.figsize": (5, 4),
    "lines.linewidth": 2,
}


# available corpora, baselines and tasks
data_choices = ["stereohoax"]
model_choices = ["all", "zeros", "ones", "random", "tfidf", "fast"]

# ---------------------------------------
# CORPORA
# ---------------------------------------

# Common labels
common = {
    "topics": [
        "xenophobia",
        "suffering",
        "economic",
        "migration",
        "culture",
        "benefits",
        "health",
        "security",
        "dehumanisation",
        "others",
    ],
    "implicit_feature": "implicit",
    "implicit_categories": [
        "Context",
        "Entailment/Evaluation",
        "Extrapolation",
        "Figures of speech",
        "Humor/Jokes",
        "Imperative/Exhortative",
        "Irony/Sarcasm",
        "World knowledge",
        "Others",
    ],
}

# StereoHoax corpus
stereohoax = {
    **common,
    "x_columns": [
        "text",
        "txt_father",
        "txt_head",
        "rh_text",
        "rh_type",
        "rh_id",
        "tweet_id",
        "conversation_id",
        "in_reply_to_tweet_id",
        "b",
        "du",
        "dd",
        "c",
        "ac",
        "p",
        "contextual",
        "implicit",
        "url",
    ],
    "indexes": ["index"],
    "text_columns": ["text", "txt_father", "txt_head", "rh_text"],
    "contexts": ["txt_father", "txt_head", "rh_text"],
    "contexts_results": [
        "0_no_context",
        "2_txt_father",
        "3_txt_head",
        "4_rh_text",
        "5_concat_contexts",
    ],
    "feature": "text",
    "target": "stereo",
    "target_pred": "stereo_pred",
    "data": "stereohoax",
    "path": "data/stereohoax",
    "datafile": "stereoHoax-ES_goldstandard.csv",
}
stereohoax["y_columns"] = [stereohoax["target"]] + stereohoax["topics"]
stereohoax["path"] = os.path.join(BASE_DIR, stereohoax["path"])


def get_conf(data: str) -> Namespace:
    """Convert the config dictionary for the wanted corpus into a Namespace"""
    if data.lower() == "stereohoax":
        conf = Namespace(**stereohoax)
    else:
        raise ValueError("Data must be 'stereohoax'")

    return conf
