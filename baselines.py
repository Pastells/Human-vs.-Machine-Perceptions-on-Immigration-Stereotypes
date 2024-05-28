"""
Create baselines
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from utils import config, io
from utils.results import DfResults

warnings.filterwarnings("ignore", category=RuntimeWarning)


def read_data(datafiles=("train.csv", "test.csv")):
    """Read train and test data and split into X and y
    y's are transformed into numpy arrays"""
    train_csv = os.path.join(conf.path, datafiles[0])
    test_csv = os.path.join(conf.path, datafiles[1])
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    return train, test


def get_X_y(train, test):
    X_train = train[conf.feature]
    y_train = train[conf.target].to_numpy()
    X_test = test[conf.feature]

    return X_train, y_train, X_test


# ------------ Models ---------------------------------


def random_classifier(X_train, y_train, X_test):
    """Weighted random classifier"""
    np.random.seed(seed=config.SEED)
    vals, prob = np.unique(y_train, return_counts=True)
    prob = prob / y_train.shape[0]
    pred = np.random.choice(vals, size=(X_test.shape[0],), p=prob)
    return pred


def svm(X_train, y_train, X_test, kernel="linear", C=1):
    """Fit and predict SVC"""
    cls = SVC(kernel=kernel, C=C, random_state=config.SEED)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)
    return pred


# ------------ Vectorizations ---------------------------------


def tfidf(X_train, X_test, ngrams=(1, 1), max_features=10000):
    """TF-IDF + SVC"""
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        ngram_range=ngrams,
        max_features=max_features,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec


def fast_text(X_trn, X_tst):
    """Get text embedding"""
    import spacy

    # Disable all components except tok2vec
    nlp = spacy.load("es_core_news_md", enable=["tok2vec"])

    def get_emb(text):
        sentences = list(nlp.pipe(text.tolist()))
        text_vec = np.array([sent.vector for sent in sentences])
        return text_vec

    X_train_vec = get_emb(X_trn)
    X_test_vec = get_emb(X_tst)

    return X_train_vec, X_test_vec


# ------------ Main ---------------------------------


def main(args):
    model = args.model

    train, test = read_data(args.datafiles)
    X_trn, y_train, X_tst = get_X_y(train, test)
    X_train = X_trn.to_numpy()
    X_test = X_tst.to_numpy()

    results = DfResults(test, conf, task=1)

    if model in ("zeros", "all"):
        results.add("all_zeros", 0, folder="baselines")

    if model in ("ones", "all"):
        results.add("all_ones", 1, folder="baselines")

    if model in ("random", "all"):
        pred = random_classifier(X_train, y_train, X_test)
        results.add("random_classifier", pred, folder="baselines")

    if model in ("tfidf", "all"):
        for ngram in (1, 3):
            X_train_vec, X_test_vec = tfidf(X_train, X_test, ngrams=(1, ngram))
            pred = svm(X_train_vec, y_train, X_test_vec)
            results.add(f"tfidf+svc_{ngram}_gram", pred, folder="baselines")

    if model in ("fast", "all"):
        logging.info("Creating embedding")
        X_train_vec, X_test_vec = fast_text(X_trn, X_tst)
        pred = svm(X_train_vec, y_train, X_test_vec)
        results.add("fast_text+svc", pred, folder="baselines")


if __name__ == "__main__":
    global conf
    args, conf = io.baselines_parsing()
    io.logging_func(args.log_print, os.path.basename(__file__))
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)

    main(args)
