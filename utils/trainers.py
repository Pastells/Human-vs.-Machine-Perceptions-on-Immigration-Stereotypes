"""
Custom optimizers and trainers
for fine-tuning BERT models with hard and soft labels

The Custom Trainers build uppon each other:
- TopKTrainer keeps the top-k best checkpoints
- SoftTrainer uses TopKTrainer
- HardTrainer uses SoftTrainer
"""

import os
import random
import shutil
import subprocess
import warnings

import numpy as np
import peft
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.training_args import ParallelMode

from utils.results import compute_metrics_hard, compute_metrics_soft

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


def memory_stats(id=""):
    print(f"Memory consumption after {id}:")
    print(f"alloc - {torch.cuda.memory_allocated() / 1024**2:.0f}Mb")
    print(f"cached - {torch.cuda.memory_cached() / 1024**2:.0f}Mb")


# ============================
# OPTIMIZERS
# ============================


def roberta_base_AdamW_LLRD(model, init_lr=3.5e-6, lr_decay=0.9, head_lr=None, **kwargs):
    """
    Layer-wise Learning Rate Decay as decribed in
    [Revisiting Few-sample BERT Fine-tuning](https://arxiv.org/abs/2006.05987)
    init_lr : initial learning_rate
    lr_decay: lr decay per layer
    head_lr: pooler and regressor lr (if not None and they exist)
    """

    # To be passed to the optimizer (only parameters of the layers you want to update).
    opt_parameters = []
    named_parameters = list(model.named_parameters())

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    lr = init_lr
    if head_lr is None:
        head_lr = lr

    # === Pooler and regressor ===

    params_0 = [
        p
        for n, p in named_parameters
        if ("pooler" in n or "regressor" in n) and any(nd in n for nd in no_decay)
    ]
    params_1 = [
        p
        for n, p in named_parameters
        if ("pooler" in n or "regressor" in n) and not any(nd in n for nd in no_decay)
    ]

    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": head_lr, "weight_decay": 0.01}
    opt_parameters.append(head_params)

    # === 12 Hidden layers ===

    for layer in range(11, -1, -1):
        params_0 = [
            p
            for n, p in named_parameters
            if f"encoder.layer.{layer}." in n and any(nd in n for nd in no_decay)
        ]
        params_1 = [
            p
            for n, p in named_parameters
            if f"encoder.layer.{layer}." in n and not any(nd in n for nd in no_decay)
        ]

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
        opt_parameters.append(layer_params)

        lr *= lr_decay

    # === Embeddings layer ===

    params_0 = [
        p for n, p in named_parameters if "embeddings" in n and any(nd in n for nd in no_decay)
    ]
    params_1 = [
        p for n, p in named_parameters if "embeddings" in n and not any(nd in n for nd in no_decay)
    ]

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": 0.01}
    opt_parameters.append(embed_params)

    return AdamW(opt_parameters, lr=init_lr, **kwargs)


def roberta_base_AdamW_grouped_LLRD(model, init_lr=1e-6, C1=1.75, C2=3.5, head_lr=None, **kwargs):
    # To be passed to the optimizer (only parameters of the layers you want to update).
    opt_parameters = []
    named_parameters = list(model.named_parameters())

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    mid_layers = ["layer.4", "layer.5", "layer.6", "layer.7"]
    end_layers = ["layer.8", "layer.9", "layer.10", "layer.11"]

    for name, params in named_parameters:
        weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01

        if "embeddings" in name or "encoder" in name:
            if any(p in name for p in mid_layers):
                lr = init_lr * C1
            elif any(p in name for p in end_layers):
                lr = init_lr * C2
            else:
                lr = init_lr

            opt_parameters.append({"params": params, "weight_decay": weight_decay, "lr": lr})

        # regressor and pooler
        if name.startswith("regressor") or "pooler" in name:
            if head_lr is None:
                head_lr = init_lr * C2

            opt_parameters.append({"params": params, "weight_decay": weight_decay, "lr": head_lr})

    return AdamW(opt_parameters, lr=init_lr, **kwargs)


# ============================
# Custom Trainers
# ============================


class TopKDict(object):
    """
    Dictionary with top-k values

    Args:
        K (`int`):
            How many values to keep in the dictionary
        greater_is_better (`bool`):
            Keep bigger values if True, else lower
        remove_folder (`bool`):
            Assumes keys are folder names. Removes folders demoted from top-k
    """

    def __init__(self, K, greater_is_better=True, remove_folder=False):
        self.dictionary = {}
        self.K = K
        self.greater_is_better = greater_is_better
        self.remove_folder = remove_folder

    def __repr__(self):
        return str(self.dictionary)

    def push(self, key, value):
        self.dictionary[key] = value
        if len(self.dictionary) > self.K:  # pop minimum (or maximum) from dict
            if self.greater_is_better:
                val = min(self.dictionary.values())
            else:
                val = max(self.dictionary.values())
            ind = list(self.dictionary.keys())[list(self.dictionary.values()).index(val)]
            self.dictionary.pop(ind, None)
            if self.remove_folder:
                shutil.rmtree(ind, ignore_errors=True)

    def __getitem__(self, key):
        return self.dictionary[key]

    def __setitem__(self, key, value):
        if self.dictionary.get(key):
            raise ValueError("key already in present")
        self.push(key, value)


class TopKTrainer(Trainer):
    def __init__(self, **kwargs):
        """
        Trainer that keeps the top-k checkpoints
        Does not rely on _rotate_checkpoints, worst checkpoints get deleted by
        TopKDict class instance `self.best_model_checkpoints`

        disclaimer: addapted from transformers==4.36.2, for use in a 4090 GPU
                    I deleted all options for fsdp, xla, npu...
        """
        super().__init__(**kwargs)
        self.best_model_checkpoints = TopKDict(
            self.args.save_total_limit,
            self.args.greater_is_better,
            self.args.should_save,
        )

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Addapted from transformers/trainer.py, check source for comments
        Uses `self.best_model_checkpoints: TopKDict`
        """
        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            # Add metric_value to TopKDict
            self.best_model_checkpoints[output_dir] = metric_value
            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # If checkpoint is not in top-k, return
        if output_dir not in self.best_model_checkpoints.dictionary:
            return

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

    def _save_rng_state(self, output_dir):
        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(
                rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth")
            )

    def _save_optimizer_and_scheduler(self, output_dir):
        if self.args.should_save:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

        # Save SCHEDULER & SCALER
        if self.args.should_save:
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)


class SoftTrainer(TopKTrainer):
    def __init__(
        self,
        LLRD=False,
        lr_decay=0.9,
        C1=1.75,
        C2=3.5,
        head_lr=None,
        num_warmup_steps=0,
        **kwargs,
    ):
        """
        Trainer for soft labels
        with cosine scheduler
        and optional Layer-wise Learning Rate Decay (LLRD)

        LLRD: if "groped" perform LLRD with 3 different LRs
                - Extra params C1 and C2, multiplicative factors
              if True perform LLRD with layer decay lr_decay
                - Extra param lr_decay
            - init_lr is the initial lr in both cases
            - head_lr as pooler and regressor lr (if not None and they exist)
                      otherwise use same as init_lr*C2 or init
        """
        super().__init__(**kwargs)
        self.LLRD = LLRD
        self.lr_decay = lr_decay
        self.C1 = C1
        self.C2 = C2
        self.head_lr = head_lr
        self.num_warmup_steps = num_warmup_steps

    def create_optimizer_and_scheduler(self, num_training_steps):
        """AdamW from pytorch"""
        if self.LLRD is False:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        elif self.LLRD == "grouped":
            self.optimizer = roberta_base_AdamW_grouped_LLRD(
                self.model,
                init_lr=self.args.learning_rate,
                C1=self.C1,
                C2=self.C2,
                head_lr=self.head_lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            self.optimizer = roberta_base_AdamW_LLRD(
                self.model,
                init_lr=self.args.learning_rate,
                lr_decay=self.lr_decay,
                head_lr=self.head_lr,
                weight_decay=self.args.weight_decay,
            )
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=num_training_steps,
        )


class HardTrainer(SoftTrainer):
    def __init__(
        self,
        class_weights=None,
        focal_loss=False,
        **kwargs,
    ):
        """
        Trainer for hard labels
        with cosine scheduler, class weights and optional focal loss

        class_weights: if not None use as weights for Cross Entropy Loss
        focal_loss: If True, use Focal Loss: https://arxiv.org/abs/1708.02002
        """
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.focal_loss = focal_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is None:
            loss_fct = torch.nn.CrossEntropyLoss().to(DEVICE)
        else:
            loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights)).to(DEVICE)

        BCEloss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        if self.focal_loss:
            gamma = 5.0
            alpha = 0.2
            pt = torch.exp(-BCEloss)  # prevents nans
            loss = alpha * (1 - pt) ** gamma * BCEloss
        else:
            loss = BCEloss

        return (loss, outputs) if return_outputs else loss


# ============================
# MAIN TRAIN FUNCTION
# ============================


def train(
    model_checkpoint,
    tok_train,
    tok_val,
    H_PARAMS: dict,
    data: str,
    name: str,
    soft_labels: bool = False,
    **trainer_args,
):
    """
    Train using HF trainer
    tok_train: tokenized train set
    tok_val: tokenized validation set
    H_PARAMS: dictionary with hyper parameters
    """
    args = TrainingArguments(
        output_dir=H_PARAMS["output_dir"],
        learning_rate=H_PARAMS["lr"],
        per_device_train_batch_size=H_PARAMS["batch_size"],
        per_device_eval_batch_size=H_PARAMS["batch_size"],
        num_train_epochs=H_PARAMS["epochs"],
        evaluation_strategy="steps",  # "epoch"
        save_strategy="steps",
        eval_steps=H_PARAMS["n_steps"],
        save_steps=H_PARAMS["n_steps"],
        logging_steps=H_PARAMS["n_steps"],
        load_best_model_at_end=True,
        save_total_limit=H_PARAMS["save"],
        save_safetensors=True,
        report_to="wandb",
        fp16=H_PARAMS["fp16"],
        # weight_decay=0.01,
        # metric_for_best_model="eval_f1",
    )

    # model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=1 if soft_labels else 2
    )

    # LoRA
    if H_PARAMS["lora"]:
        args.warmup_ratio = 0.1
        args.max_grad_norm = 0.3
        peft_config = peft.LoraConfig(
            task_type=peft.TaskType.SEQ_CLS,
            r=2,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
        )
        model = peft.get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # trainer
    trainer_args = {
        "model": model,
        "args": args,
        "train_dataset": tok_train,
        "eval_dataset": tok_val,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=H_PARAMS["patience"])],
        **trainer_args,
    }
    if soft_labels:
        trainer = SoftTrainer(compute_metrics=compute_metrics_soft, **trainer_args)
    else:
        if H_PARAMS["class_weights"]:
            class_weights = compute_class_weight(
                "balanced",
                classes=[0, 1],
                y=tok_train["labels"].tolist(),
            ).astype("float32")
        else:
            class_weights = None

        trainer = HardTrainer(
            class_weights=class_weights,
            compute_metrics=compute_metrics_hard,
            **trainer_args,
        )

    # train
    trainer.train()

    # Rename best checkpoint folder
    if H_PARAMS["save_models"]:
        try:
            checkpoint = os.path.basename(trainer.state.best_model_checkpoint)
            print(f"Best checkpoint: {checkpoint}")
            subprocess.run(
                [
                    "cp",
                    f"{checkpoint}/model.safetensors",
                    f"models/{data}/{name}.safetensors",
                ]
            )
        except TypeError:
            print("No checkpoint found")

    return trainer, model
