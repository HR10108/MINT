#!/usr/bin/env python
# coding=utf-8
import logging
import os
import random
import sys
import time

import datasets
import torch
import torch.distributed as dist
import transformers

# from instruction_tuning.train.lora_trainer import LoRAFSDPTrainer, Trainer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
    set_seed,
)
import torch.nn.functional as F
from mint.get_grads.get_training_dataset import get_training_dataset
from mint.train.data_arguments import DataArguments, get_data_statistics
from mint.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from mint.train.training_arguments import TrainingArguments
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


IGNORE_INDEX = -100  # 确保您有定义这个常量，通常用于忽略的标签


def add_weights_to_dataset(dataset, weights):
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Weights size: {len(weights)}")

    def add_weight(example, idx):
        example["weight"] = weights[idx]
        return example

    # 使用 map 给每个样本添加 weight 字段
    dataset = dataset.map(add_weight, with_indices=True)

    logger.info(f"Dataset with weights: {dataset[0]['weight']}")
    return dataset


def weighted_ce_loss(
    outputs,
    labels,
    weights=None,  # shape: (B,)
    ignore_index: int = IGNORE_INDEX,
):
    """
    Token-level平均 + 样本权重 的交叉熵损失。

    参数
    ----
    outputs : ModelOutput / dict，必须含 logits (B, L, V)
    labels  : LongTensor (B, L)  目标 token 序列，忽略值 = ignore_index
    weights : FloatTensor (B,)   每条样本的权重；若为 None，则等价于全 1
    """
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()  # (B, L-1, V)
    shift_labels = labels[..., 1:].contiguous()  # (B, L-1)

    # 1. 逐 token 交叉熵（不做 reduce）
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view_as(
        shift_labels
    )  # (B, L-1)

    # 2. 有效 token mask
    valid_mask = shift_labels.ne(ignore_index)  # (B, L-1)

    # 3. 准备并 broadcast 样本权重
    if weights is None:
        weights = torch.ones(
            shift_labels.size(0), device=shift_labels.device, dtype=token_loss.dtype
        )
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(
            weights, device=shift_labels.device, dtype=token_loss.dtype
        )
    weights = weights.view(-1, 1)  # (B, 1)
    weight_mask = weights * valid_mask  # broadcast → (B, L-1)

    # 4. 按权重汇总，再按 *总 token 数* 归一化
    numerator = (token_loss * weight_mask).sum()
    denominator = weight_mask.sum().clamp(min=1.0)  # 总有效 token *权重
    loss = numerator / denominator
    return loss


def weighted_loss_function(outputs, labels, weights=None):
    """
    Computes a weighted cross-entropy loss.
    """
    logits = (
        outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    )  # (B, L, V)

    shift_logits = logits[..., :-1, :].contiguous()  # (B, L-1, V)
    shift_labels = labels[..., 1:].contiguous()  # (B, L-1)

    # Cross entropy loss without reduction to calculate loss for each token
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view(
        shift_labels.size()
    )  # (B, L-1)

    # Create a valid mask to ignore IGNORE_INDEX tokens
    valid_mask = shift_labels.ne(IGNORE_INDEX)  # (B, L-1)
    tokens_per_row = valid_mask.sum(dim=1).clamp(min=1)  # To avoid division by zero

    # Compute the per sample loss for valid tokens only
    per_sample_loss = (token_loss * valid_mask).sum(dim=1) / tokens_per_row  # (B,)

    # Apply weights if provided
    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(
                weights, dtype=per_sample_loss.dtype, device=per_sample_loss.device
            )
        weights = weights.view(-1)  # Ensure it has the correct shape (B,)
        per_sample_loss = per_sample_loss * weights  # Apply weight to per sample loss
    # Return the mean loss across all samples in the batch
    return per_sample_loss.mean()


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight", None)  # (B,)
        labels = inputs.get("labels")
        outputs = model(**inputs)

        loss = weighted_ce_loss(outputs, labels, weights)
        return (loss, outputs) if return_outputs else loss


def load_coreset(coreset_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the coreset file and return the subset and weights."""
    coreset_data = np.load(coreset_file)
    subset = coreset_data["subset"]
    weights = coreset_data["weights"]
    return subset, weights


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # Load the Coreset file if provided
    if data_args.coreset_file:
        logger.info(f"Loading Coreset file: {data_args.coreset_file}")
        subset, weights = load_coreset(data_args.coreset_file)
        logger.info(f"Loaded Coreset with {len(subset)} samples.")
        weights = weights / np.mean(weights)  # Normalize weights
        logger.info(f"Normalized weights: {weights[:10]}...")
        logger.info(f"Subset indices: {subset[:10]}...")
        # Load the full training dataset (we will sample based on the coreset)
        train_dataset = get_training_dataset(
            data_args.train_files,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            sample_percentage=data_args.percentage,
            seed=data_args.sample_data_seed,
        )

        # Use the subset indices to filter the training dataset
        logger.info(f"Filtering the dataset based on the Coreset subset...")
        logger.info(f"Original dataset size: {len(train_dataset)}")
        train_dataset = train_dataset.select(subset)  # Use the subset of data
        logger.info(f"Filtered dataset size: {len(train_dataset)}")
        # 假设你已经加载了 weights
        train_dataset = add_weights_to_dataset(train_dataset, weights)

        # 2. 确保移除多余的列，保留需要的列
        required_columns = ["input_ids", "labels", "attention_mask", "weight"]
        cols_to_remove = [
            col for col in train_dataset.column_names if col not in required_columns
        ]

        # 移除多余的列
        train_dataset = train_dataset.remove_columns(cols_to_remove)

        # 检查更新后的数据集
        logger.info(train_dataset.column_names)
        logger.info(f"Sample from train dataset: {train_dataset[0]}")

    else:
        # If no Coreset is provided, fall back to the full dataset
        train_dataset = get_training_dataset(
            data_args.train_files,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            sample_percentage=data_args.percentage,
            seed=data_args.sample_data_seed,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=model_args.torch_dtype
    )
    add_padding_to_tokenizer(tokenizer)

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False

    if not isinstance(model, PeftModel) and model_args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"Applied LoRA to model.")
        model.print_trainable_parameters()

        # for checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    get_data_statistics(train_dataset)

    if "dataset" in train_dataset.features:
        train_dataset = train_dataset.remove_columns(["dataset", "id", "messages"])

    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")

    analysis_dataset = None
    if training_args.analysis_mode:
        from mint.get_grads.get_validation_dataset import get_dataset

        analysis_dataset = get_dataset(
            training_args.analysis_dataset,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            max_length=data_args.max_seq_length,
        )

    if dist.is_initialized() and dist.get_rank() == 0:
        print(model)
    elif not dist.is_initialized():
        print(model)

    assert data_args.coreset_file, "Coreset file is required for training."
    training_args.remove_unused_columns = False
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=analysis_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest"
        ),
    )

    # Training
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # remove the full model in the end to save space, only adapter is needed
    if isinstance(model, PeftModel):
        pytorch_model_path = os.path.join(
            training_args.output_dir, "pytorch_model_fsdp.bin"
        )
        os.remove(pytorch_model_path) if os.path.exists(pytorch_model_path) else None


if __name__ == "__main__":
    main()
