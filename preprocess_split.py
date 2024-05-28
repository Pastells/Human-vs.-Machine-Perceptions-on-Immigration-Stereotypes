"""
Preprocess StereoHoax corpus and split into train, (validation) and test
"""

import itertools
import logging
import os
import re
import string

import nltk
import numpy as np
import pandas as pd
from emoji import demojize
from skmultilearn.model_selection import IterativeStratification

from utils import config, io

# ------------------- Global variables ---------------------

stopwords = set(nltk.corpus.stopwords.words("spanish"))

# map punctuation (except @) to space
PUNCTUATION = string.punctuation.replace("@", "") + "¿?¡…"
punctuation_to_space = str.maketrans(PUNCTUATION, " " * len(PUNCTUATION))


def import_spacy():
    """Spacy with custom lemmas"""
    import spacy

    nlp = spacy.load("es_core_news_md", disable=["ner"])
    nlp.get_pipe("attribute_ruler").add([[{"TEXT": "URL"}]], {"LEMMA": "URL"})
    nlp.get_pipe("attribute_ruler").add([[{"TEXT": "NUM"}]], {"LEMMA": "NUM"})
    nlp.get_pipe("attribute_ruler").add([[{"TEXT": "jaja"}]], {"LEMMA": "jaja"})
    return nlp


# -----------------------------------------------------------------


def masking(text: str, user="@user", url="URL", num="NUM", laughter="jaja") -> str:
    """Mask usernames, URLs, numbers and laughter with the given (or default) tokens"""
    text = re.sub(r"@[\w]{2,}", user, text)
    text = re.sub(r"http\S+", url, text)

    # split numbers and letters with a space (also the euro symbol "€")
    text = re.sub(r"([A-Za-zÑñÇçÀàÁáÈèÉéÍíÏïÒòÓóÚúÜü])(\d+)", r"\1 \2", text)
    text = re.sub(r"(\d+)([A-Za-zÑñÇçÀàÁáÈèÉéÍíÏïÒòÓóÚúÜü€])", r"\1 \2", text)
    # mask with num token
    text = re.sub(r"(\d+\.)?\d+(,\d+)?", num, text)

    text = re.sub(r"a?j+a+j\w+", laughter, text)
    text = re.sub(r"a?h+a+h\w+", laughter, text)
    text = re.sub(r"text+d+", laughter, text)
    text = re.sub(r"[:;]\)", laughter, text)
    return text


def process_hashtags(text, hashtag_token="hashtag") -> str:
    """Separate hashtags into hashtag_token and de-camelized tokens"""

    text = text.groups()[0]

    # Convert camel case to different words
    start_of_camel = re.compile(r"([A-Z]+)")
    text = start_of_camel.sub(r" \1", text).strip()

    text = hashtag_token + " " + text
    return text


def preprocess(
    text: str, emoji_wrapper="emoji", min_len: int = 1, shorten: int = 2, lower=True
) -> list:
    """Preprocess before tokenization
    min_len: Remove all tokens with length <= min_len
    shorten:
    emoji_wrapper: token before and after emoji
    """

    # remove short words
    text = re.sub(r"\b\w{1,%d}\b" % min_len, "", text)

    # remove repeated characters
    repeated_regex = re.compile(r"(.)" + r"\1" * (shorten - 1) + "+")
    text = repeated_regex.sub(r"\1" * shorten, text)

    # hashtags
    hashtag_regex = re.compile(r"\B#(\w*[a-zA-Z]+\w*)")
    text = hashtag_regex.sub(process_hashtags, text)

    # lower
    if lower:
        text = text.lower()

    # mask usernames, URLs, numbers and laughter
    text = masking(text)

    # change punctuation, tabs and new lines into spaces
    text = text.translate(punctuation_to_space)
    text = re.sub(r"[\n\t]", " ", text)

    # translate emoji
    delim = f" {emoji_wrapper} "
    text = demojize(text, language="es", delimiters=(delim, delim))
    text = re.sub(r"_", " ", text)

    # removes extra spaces
    text = re.sub(r"\s\s+", " ", text.strip())

    return text


def tokenize(ds: pd.Series, nlp, lemma=True):
    """Tokenize or lemmatize
    lemma: Output lemma if True
                   token if False
    """
    sentences = list(nlp.pipe(ds.tolist()))
    if lemma:
        text = [
            list(
                itertools.chain.from_iterable(
                    [t.lemma_.split(" ") for t in sent if t.text not in stopwords]
                )
            )
            for sent in sentences
        ]
    else:
        text = [[t.text for t in sent if t.text not in stopwords] for sent in sentences]
    return text


def clean_data(data, lemma=True, pre=True, mask=False) -> pd.DataFrame:
    """
    + Read data
    + Clean unclassified
    + Apply preprocess to text columns

    lemma:  Use lemma if True
                 token if False

    pre: preprocess if True
    mask: apply only mask if True (and pre is False)
    """
    df = pd.read_csv(data)
    ll = len(df)
    df.dropna(inplace=True)
    assert ll == len(df)

    # Remove unclassified with stereotypes. TODO: classify
    # len0 = len(df)
    # df = df[~((df[conf.target] == 1) & (df[conf.topics] == 0).all(1))]
    # logging.info("Removed %d unclassified with stereotypes", len0 - len(df))

    if pre:
        nlp = import_spacy()
        for column in conf.text_columns:
            logging.info("Preprocessing %s", column)
            df[column] = df[column].apply(preprocess)
            df[column] = tokenize(df[column], nlp, lemma=lemma)
    elif mask:
        for column in conf.text_columns:
            df[column] = df[column].apply(masking)

    return df


# -----------------------------------------------------------------


def get_stratified_split(
    df: pd.DataFrame, labels: list, test_ratio=0.2, val_ratio=None
) -> np.ndarray:
    """Train/test split preserving distribution of stereotypes and topics"""
    stratifier = IterativeStratification(
        n_splits=2,
        order=2,
        sample_distribution_per_fold=[test_ratio, 1.0 - test_ratio],
    )
    train_ixs, test_ixs = next(stratifier.split(df[conf.feature].to_numpy(), df[labels].values))
    split_ixs = np.zeros((df.shape[0],), dtype=np.int8)
    split_ixs[test_ixs] += 1
    if val_ratio is not None:
        val_ratio = val_ratio / (1 - test_ratio)
        split_ixs[train_ixs] += get_stratified_split(
            df.iloc[train_ixs],
            labels,
            val_ratio,
        )
        split_ixs[test_ixs] += 1
    return split_ixs


def check_split(label, train, val, test):
    """Print number and proportion of each topic for
    both train, val and test"""

    count_train = train.groupby(label)["others"].count()
    count_val = val.groupby(label)["others"].count()

    ratio_train = count_train[1] / count_train[0] * 100
    ratio_val = count_val[1] / count_val[0] * 100

    if len(test) > 0:
        count_test = test.groupby(label)["others"].count()
        ratio_test = count_test[1] / count_test[0] * 100
    else:
        ratio_test = 0

    logging.info(
        "\t %-20s %2.2f %2.2f %2.2f",
        label,
        ratio_train,
        ratio_val,
        ratio_test,
    )


def split_data(df: pd.DataFrame, pre=True, lemma=True, test_ratio=0.2, val_ratio=False):
    """Split and save data
    if pre is True save as train.csv and test.csv
    if pre is False (no preprocessing) save as
        no_pre_train.csv and no_pre_test.csv"""

    # Shuffle data
    df = df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)

    split = get_stratified_split(df, conf.y_columns, test_ratio, val_ratio)

    train = df.iloc[np.argwhere(split == 0).squeeze()]
    val = df.iloc[np.argwhere(split == 1).squeeze()]
    test = df.iloc[np.argwhere(split == 2).squeeze()]

    logging.info("Ratios for train, val and test:")
    for label in conf.y_columns:
        check_split(label, train, val, test)

    save_data(train, pre, lemma, "train.csv")
    if val_ratio:
        save_data(val, pre, lemma, "val.csv")
        save_data(test, pre, lemma, "test.csv")
    else:
        save_data(val, pre, lemma, "test.csv")


def save_data(df: pd.DataFrame, pre=True, lemma=True, name: str = "clean.csv"):
    """Save data"""
    if pre:
        no_pre = ""
        tok = "" if lemma else "tok_"
    else:
        no_pre = "no_pre_"
        tok = ""

    file = os.path.join(conf.path, no_pre + tok + name)
    df.to_csv(file, index=False)
    logging.info(f"Created {file}")


# -----------------------------------------------------------------


def main():
    global conf
    args, conf = io.preprocess_parsing()
    io.logging_func(args.log_print, os.path.basename(__file__))
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)

    data = os.path.join(conf.path, conf.datafile)
    df = clean_data(data, lemma=args.lemma, pre=args.pre, mask=args.mask)

    if args.split:
        split_data(df, args.pre, args.lemma, args.test_ratio, args.val_ratio)
    else:
        save_data(df, args.pre, args.lemma)


if __name__ == "__main__":
    main()
