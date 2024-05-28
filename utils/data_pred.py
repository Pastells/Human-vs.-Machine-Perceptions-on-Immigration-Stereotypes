"""
Utils used for fine-tuning BERT models with hard and soft labels:
- Data and processing
- lr fixed to 1e-5 for soft_labels and 2e-5 for hard_labels
- Predictions
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset
from transformers import AutoTokenizer

from utils import config
from utils.results import DfResults

# ============================
# DATA and PROCESSING
# ============================


def get_data(
    data="stereohoax",
    context=None,
    append=False,
    fill=True,
    cont_same_as_txt=True,
    fill_same=True,
    soft_labels=False,
    files=None,
):
    """
    return shuffled data

    data: stereohoax
    context:
        - Stereohoax: None, txt_father, txt_head or rh_text
    append: if True, prepend context to text
            if False, use context as parent_text -> BERT [SEP] token
    fill: if True, fill missing contexts with ones one level above (done beforehand)
          if False, use 'no_fill' contexts files
    cont_same_as_txt: (in case fill is True), if True allow context to be equal to text
    fill_same: (in case cont_same_as_txt is False), if True fill the context with next
               level; if False, drop.
    soft_labels: return soft_labels if True, otherwise hard
    files: if not None, the given files are used instead of the default ones
    """
    data = data.lower()
    no_fill = "_no_fill" if fill is False else ""
    if data == "stereohoax":
        conf = config.get_conf("stereohoax")
        if files is None:
            files = (
                f"train_split_context{no_fill}_soft.csv",
                f"val_split_context{no_fill}_soft.csv",
                f"test_split_context{no_fill}_soft.csv",
            )
    else:
        raise ValueError("Invalid data")

    files = [os.path.join(conf.path, file) for file in files]

    features = conf.feature
    labels = conf.target
    if soft_labels:
        labels += "_soft"

    def get_split(
        file,
        context=None,
        append=False,
        fill=True,
        cont_same_as_txt=True,
        fill_same=True,
    ):
        split = pd.read_csv(file).fillna("")

        # 0) debug with result in context
        if context == "debug":
            split["debug"] = np.where(split[labels] == 1, "con estereotipo", "sin estereotipo")

        # a) no original context -> drop
        elif context and not fill:
            split = split.drop(split[split[context].isin(("", "0", "[]"))].index).dropna()

        # b) context is the same as text -> drop
        elif context and not cont_same_as_txt and not fill_same:
            split = split.drop(split.query(f"text == {context}").index).dropna()

        # c) context is the same as text -> fill
        elif context and not cont_same_as_txt and fill_same:
            mask = split["text"] == split[context]
            next_cont_ind = conf.contexts.index(context) + 1
            while mask.any():
                split.loc[mask, context] = split.loc[mask, conf.contexts[next_cont_ind]]
                mask = split["text"] == split[context]
                next_cont_ind += 1

        # d) fill context (done beforehand)
        elif context and fill:
            pass

        if context and append:
            split[features] = split[context] + " " + split[features]
            columns = {features: "text", labels: "labels"}
        elif context:
            columns = {features: "text", labels: "labels", context: "parent_text"}
        else:
            columns = {features: "text", labels: "labels"}

        split = split.sample(frac=1, random_state=config.SEED).reset_index(drop=True)  # shuffle
        split_set = split[columns.keys()].rename(columns=columns)
        return split, split_set

    args = {
        "context": context,
        "append": append,
        "fill": fill,
        "cont_same_as_txt": cont_same_as_txt,
        "fill_same": fill_same,
    }
    _, train_set = get_split(files[0], **args)
    _, val_set = get_split(files[1], **args)
    test, test_set = get_split(files[2], **args)

    return conf, test, train_set, val_set, test_set


def get_lr(soft_labels):
    if soft_labels:
        return 1e-5
    return 2e-5


def get_contexts(data, CONTEXTS) -> tuple:
    """
    1. CONTEXTS == "all" -> return all contexts for `data`
    2. CONTEXTS == ("all", ...) -> all contexts except ...
    3. CONTEXTS == (...,) returns CONTEXTS
    """
    exceptions = tuple()
    if type(CONTEXTS) == tuple and CONTEXTS[0] == "all":
        exceptions = CONTEXTS[1:]
    elif CONTEXTS != "all":
        return CONTEXTS

    if data == "stereohoax":
        cont = (
            None,
            "txt_father",
            "txt_head",
            "rh_text",
            "concat_contexts",
        )
    else:
        raise ValueError("wrong data")

    return tuple(c for c in cont if c not in exceptions)


def preprocess(model_checkpoint, train_set, val_set, test_set, tokenizer=None, max_tokens=128):
    """
    - tokenization from pretrained model
    - preprocessing for Robertuito model

    tokenizer: either give already loaded model or model_checkpoint to load
    """
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    pre = True if (model_checkpoint == "pysentimiento/robertuito-base-uncased") else False
    if pre:
        from pysentimiento.preprocessing import preprocess_tweet

    tokenize_args = {
        "padding": "max_length",
        "truncation": True,
        "max_length": max_tokens,
    }

    def tokenize(examples):
        tokenized_inputs = tokenizer(examples["text"], **tokenize_args)
        return tokenized_inputs

    def tokenize_context(examples):
        tokenized_inputs = tokenizer(examples["text"], examples["parent_text"], **tokenize_args)
        return tokenized_inputs

    def preprocessing_data(df, pre=False):
        if pre:
            df["text"] = df["text"].apply(preprocess_tweet)
            if "parent_text" in df.columns:
                df["parent_text"] = df["parent_text"].apply(preprocess_tweet)

        dt = Dataset.from_pandas(df)

        if "parent_text" in df.columns:
            tokenized_dt = dt.map(
                tokenize_context, remove_columns=["text", "parent_text"], batched=True
            )
        else:
            tokenized_dt = dt.map(tokenize, remove_columns=["text"], batched=True)

        return tokenized_dt.with_format("torch")

    tok_train = preprocessing_data(train_set, pre=pre)
    tok_val = preprocessing_data(val_set, pre=pre)
    tok_test = preprocessing_data(test_set, pre=pre)

    return tok_train, tok_val, tok_test


# ============================
# PREDICTIONS
# ============================


def predict(trainer, tok_test, results: DfResults, name, soft_labels, log=True):
    """
    Wrapper for soft and hard labels predictions from trainer
    adds results to DfResults instance (creating a csv) and wandb log
    """
    if soft_labels:
        pred, pred_soft = predict_soft(trainer, tok_test)
        results.add(name, pred, pred_soft)
    else:
        pred_labels, softmax_confidence = predict_hard(trainer, tok_test)
        results.add(name, pred=pred_labels, softmax_confidence=softmax_confidence)

    if log:
        wandb.log(
            {
                f"test/{col}": results.df.iloc[-1][col]
                for col in results.df.columns
                if results.df.iloc[-1][col] != 0 and col != "model"
            }
        )


def logits_to_confidence(logits, scale=False):
    """
    scale: re-scale confidence to [0,1]
    """
    probs = F.softmax(logits, dim=1)
    max = probs.topk(2, dim=1)
    pred = max.indices[:, 0].cpu().numpy()
    max = max.values
    max1 = max[:, 0]

    if scale:
        max2 = max[:, 1]
        # For more than 2 possible labels:
        # softmax_confidence = ((max1 - max2) / torch.abs(max1 + max2)).cpu().numpy()
        softmax_confidence = (max1 - max2).cpu().numpy()
    else:
        softmax_confidence = max1.cpu().numpy()

    return pred, softmax_confidence


def predict_hard(trainer, tok_test, scale=False):
    """
    hard_labels prediction and softmax_confidence from trainer
    """
    pred = trainer.predict(tok_test)
    logits = torch.tensor(pred[0])
    pred, softmax_confidence = logits_to_confidence(logits, scale)

    return pred, softmax_confidence


def predict_soft(trainer, tok_test):
    """
    soft_labels prediction from trainer
    """
    predictions = trainer.predict(tok_test)
    pred_soft = predictions[0]
    pred = np.where(pred_soft > 0.5, 1, 0)
    return pred, pred_soft


def predict_from_saved(
    model,
    tokenizer,
    test_set,
    device,
    state_dict=None,
    max_tokens: int = 128,
    batch_size: int = 32,
    temperature: float = 1,
    scale=False,
):
    if state_dict:
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # Move the model inputs to the CUDA device
    test_inputs = tokenizer.batch_encode_plus(
        test_set["text"].tolist(),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
    )
    test_inputs = {k: v.to(device) for k, v in test_inputs.items()}

    # Perform inference on the CUDA device in batches
    num_batches = len(test_inputs["input_ids"] - 1) // batch_size + 1
    logits_list = []
    with torch.no_grad():
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(test_inputs["input_ids"]))
            batch_inputs = {k: v[batch_start:batch_end] for k, v in test_inputs.items()}
            result = model(**batch_inputs)
            logits = result["logits"]
            logits_list.append(logits)
        logits = torch.cat(logits_list).to(device) / temperature

    pred, softmax_confidence = logits_to_confidence(logits, scale)

    del model
    torch.cuda.empty_cache()
    return pred, logits, softmax_confidence
