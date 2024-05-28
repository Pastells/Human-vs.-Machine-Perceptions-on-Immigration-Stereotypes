import gc
import os
import random
import shutil

import numpy as np
import torch
import wandb
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from utils.data_pred import get_contexts, get_data, get_lr, predict, preprocess
from utils.results import DfResults
from utils.trainers import memory_stats, train


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def number_contexts(x):
    x = x.replace("-None", "-0_no_context")
    x = x.replace("-concat_contexts", "-5_concat_contexts")

    x = x.replace("-previous_sentences", "-1_previous_sentences")
    x = x.replace("-previous_comment", "-2_previous_comment")
    x = x.replace("-first_comment", "-3_first_comment")
    x = x.replace("-news_title", "-4_news_title")

    x = x.replace("-txt_father", "-2_txt_father")
    x = x.replace("-txt_head", "-3_txt_head")
    x = x.replace("-rh_text", "-4_rh_text")
    return x


def rename_move(data, colab=False, save_models=False):
    if colab:
        folder = "experiments_stereohoax"
    else:
        folder = ""

    results = os.path.join(folder, "results", data)
    models = os.path.join(folder, "models", data)

    for f in os.listdir(results):
        if f.endswith(".csv"):
            f = os.path.join(results, f)
            f_new = f.replace("-task1-task1.csv", "-task1.csv")
            f_new = number_contexts(f_new)
            os.rename(f, f_new)

    for f in os.listdir(models):
        if f.endswith(".safetensors"):
            f = os.path.join(results, f)
            f_new = f.replace("None", "no_context")
            os.rename(f, f_new)

    if not colab:
        return

    if not os.path.isdir("drive/MyDrive"):
        print("Drive not mounted")
        return

    # Move CSV files
    csv_source_dir = os.path.join("experiments_stereohoax", "results", data)
    csv_destination_dir = os.path.join("drive", "MyDrive", "fine_tuning", "results", data)

    for filename in os.listdir(csv_source_dir):
        if filename.endswith(".csv"):
            csv_source_file = os.path.join(csv_source_dir, filename)
            csv_destination_file = os.path.join(csv_destination_dir, filename)
            shutil.move(csv_source_file, csv_destination_file)

    # Move model files if save_models is True
    if save_models:
        models_source_dir = os.path.join("models", data)
        models_destination_dir = os.path.join("drive", "MyDrive", "fine_tuning", "models", data)

        for filename in os.listdir(models_source_dir):
            models_source_file = os.path.join(models_source_dir, filename)
            models_destination_file = os.path.join(models_destination_dir, filename)
            shutil.move(models_source_file, models_destination_file)


def main_loop(SEEDS_DATA_SOFT, DATA_PARAMS, CONTEXTS, H_PARAMS, MODELS, GLOBALS, LLRD):
    for seed, (data, soft_labels) in tqdm(SEEDS_DATA_SOFT, desc="data loop", position=0):
        DATA_PARAMS["data"] = data
        DATA_PARAMS["soft_labels"] = soft_labels
        H_PARAMS["lr"] = get_lr(soft_labels)
        conf, test, _, _, _ = get_data(**DATA_PARAMS)
        data_contexts = get_contexts(data, CONTEXTS)
        hard_or_soft = "hard" if soft_labels is False else "soft"
        results = DfResults(test, conf)

        for model_name, model_checkpoint in tqdm(
            MODELS.items(), desc="models loop", position=1, leave=False
        ):
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

            for context in tqdm(data_contexts, desc="context loop", position=2, leave=False):
                set_deterministic(seed=seed)
                DATA_PARAMS["context"] = context
                _, _, train_set, val_set, test_set = get_data(**DATA_PARAMS)
                # train_set, val_set, test_set = add_noise_splits(
                #     train_set, val_set, test_set
                # )

                tok_train, tok_val, tok_test = preprocess(
                    model_checkpoint,
                    train_set,
                    val_set,
                    test_set,
                    max_tokens=GLOBALS["max_tokens"],
                    tokenizer=tokenizer,
                )

                name = (
                    f"{data}-{model_name}-{context}_{hard_or_soft}_s{seed}"
                    + f"_{GLOBALS['experiment_name']}-task1"
                )
                name = number_contexts(name)
                print(name)
                if (os.path.isfile(f"results/{data}/" + name + ".csv")) or (
                    os.path.isfile(f"results/{data}/fine_tuning_512/" + name + ".csv")
                    or os.path.isfile(
                        f"/home/ppastells/projects/experiments_stereohoax/results/{data}/fine_tuning_512/"
                        + name
                        + ".csv"
                    )
                    or os.path.isfile(
                        f"/home/ppastells/projects/experiments_stereohoax/results/{data}/"
                        + name
                        + ".csv"
                    )
                    or os.path.isfile(
                        f"/home/ppastells/projects/experiments_stereohoax_cp/results/{data}/fine_tuning_512/"
                        + name
                        + ".csv"
                    )
                    or os.path.isfile(
                        f"/home/ppastells/projects/experiments_stereohoax_cp/results/{data}/"
                        + name
                        + ".csv"
                    )
                ):
                    print("\t already exists")
                    continue

                wandb.init(
                    project="experiments_stereohoax",
                    group=GLOBALS["experiment_name"],
                    config={
                        "name": name,
                        "model_name": model_name,
                        **DATA_PARAMS,
                        **LLRD,
                    },
                )
                trainer, model = train(
                    model_checkpoint=model_checkpoint,
                    tok_train=tok_train,
                    tok_val=tok_val,
                    data=data,
                    name=name,
                    H_PARAMS=H_PARAMS,
                    soft_labels=soft_labels,
                    **LLRD,
                )
                torch.cuda.empty_cache()
                memory_stats("train")

                predict(trainer, tok_test, results, name, soft_labels)

                rename_move(data, GLOBALS["colab"], save_models=H_PARAMS["save_models"])

                del model
                del trainer
                torch.cuda.empty_cache()
                gc.collect()
                memory_stats("run")
                wandb.finish(quiet=True)
