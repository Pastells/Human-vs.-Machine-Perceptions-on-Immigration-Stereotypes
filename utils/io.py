"""Inputs and outputs:
   + parsers for both preprocess_split.py and baselines.py
   + logging
   + addapted to calls from jupyter notebooks
"""

import argparse
import logging
import os
import sys
from datetime import datetime

from utils import config

# --------- Helper functions ----------


def str2bool(val):
    """Convert a string representation of truth to true (1) or false (0)"""
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# --------- Parsing ----------


def preprocess_parsing(jupyter=None):
    """Parse input arguments and return them in args object
    From a jupyter notebook call as `parsing("")`"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""
        Preprocess corpus and split into train, (validation) and test
        Resulting train.csv, (val.csv) and test.csv are saved in the data folder""",
    )
    parser.add_argument(
        "-data",
        type=str,
        default="stereohoax",
        choices=config.data_choices,
        help="Input corpus",
    )
    parser.add_argument(
        "-datafile",
        type=str,
        default="",
        help="Data file. Default is the one in config.py",
    )
    parser.add_argument(
        "-sep",
        type=str,
        default="",
        help="Data file separator. Default is the one in config.py",
    )
    parser.add_argument(
        "-lemma",
        type=str2bool,
        default=True,
        help="Use lemmas if True, otherwise tokens",
    )
    parser.add_argument(
        "-pre",
        type=str2bool,
        default=True,
        help="Preprocess data",
    )
    parser.add_argument(
        "-mask",
        type=str2bool,
        default=True,
        help="apply only mask if True (and pre is False)",
    )
    parser.add_argument(
        "-split",
        type=str2bool,
        default=False,
        help="Split data",
    )
    parser.add_argument(
        "-test_ratio",
        type=float,
        default=0.2,
        help="Test ratio (0,1)",
    )
    parser.add_argument(
        "-val_ratio",
        type=float,
        default=None,
        help="""Validation ratio (0,1)
        If not specified the resulting files are train and test only.""",
    )
    parser.add_argument("-log_print", action="store_true", help="Print all log output")

    args = parser.parse_args(jupyter)
    conf = config.get_conf(args.data)

    if args.sep:
        conf.sep = args.sep

    if args.datafile:
        conf.datafile = args.datafile

    return args, conf


def baselines_parsing(jupyter=None):
    """Parse input arguments and return them in args object
    From a jupyter notebook call as `parsing("")`"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create baselines",
    )
    parser.add_argument(
        "-data",
        type=str,
        default="stereohoax",
        choices=config.data_choices,
        help="Input corpus",
    )
    parser.add_argument(
        "-datafiles",
        type=str,
        nargs=2,
        default=["train.csv", "test.csv"],
        help="Train and test files",
    )
    parser.add_argument(
        "-model",
        type=str,
        default="all",
        choices=config.model_choices,
        help="Model to run",
    )
    parser.add_argument(
        "-append",
        type=str2bool,
        default=True,
        help="""Append to metrics files in results.""",
    )
    parser.add_argument("-log_print", action="store_true", help="Print all log output")

    args = parser.parse_args(jupyter)
    conf = config.get_conf(args.data)

    return args, conf


# --------- Logging ----------


def logging_func(log_print: bool, name: str):
    """Create logger
    log_print: print logging if is TRUE
    name: name of the logging file (+ .log)
    """

    # check if log file already existed
    name = datetime.now().strftime(f"{name[:-3]}_%Y-%m-%d_%H:%M:%S.log")
    format_str = "[%(asctime)s - %(levelname)s] %(message)s"

    logging.basicConfig(
        filename=os.path.join(config.LOGS_DIR, name),
        format=format_str,
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Print log info
    if log_print:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
