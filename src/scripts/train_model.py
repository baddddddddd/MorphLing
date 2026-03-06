import math
import os
import re

import hydra
import torchinfo
from datasets import load_dataset
from huggingface_hub import login, HfApi, snapshot_download
from omegaconf import DictConfig, OmegaConf
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from ..tokenizers import MorphlingTokenizer, SentencePieceTokenizer


def calculate_intermediate_size(hidden_size: int) -> int:
    MULTIPLE_OF = 256

    intermediate_size = int(8 * hidden_size / 3)
    intermediate_size = MULTIPLE_OF * (
        (intermediate_size + MULTIPLE_OF - 1) // MULTIPLE_OF
    )
    return intermediate_size


def calculate_num_hidden_layers(hidden_size: int) -> int:
    return math.ceil(hidden_size / 64)


def calculate_num_attention_heads(hidden_size: int) -> int:
    return math.ceil(hidden_size / 64)


tokenizer_registry = {
    "MorphlingTokenizer": MorphlingTokenizer,
    "SentencePieceTokenizer": SentencePieceTokenizer,
}


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print("=== Active Configuration ===")
    print(OmegaConf.to_yaml(cfg))

    if "hf_token" not in cfg:
        raise Exception(
            "hf_token is required, add +hf_token=YOUR_TOKEN_HERE when running command"
        )

    login(cfg.hf_token)

    api = HfApi()
    user = api.whoami()
    username = user["name"]

    print(f"\n> Logged in as {username}")

    TokenizerClass = tokenizer_registry[cfg.tokenizer.name]
    tokenizer = TokenizerClass(cfg.tokenizer.file)

    print("\n=== LLaMa Configuration ===")
    hidden_size = cfg.model.hidden_size
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        intermediate_size=calculate_intermediate_size(hidden_size),
        num_hidden_layers=calculate_num_hidden_layers(hidden_size),
        num_attention_heads=calculate_num_attention_heads(hidden_size),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,
        max_position_embeddings=cfg.model.context_window,
    )

    print(f"  hidden_size: {config.hidden_size}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  tie_word_embeddings: {config.tie_word_embeddings}")

    model = LlamaForCausalLM(config)

    print()
    torchinfo.summary(model)

    print(f"\n> Loading dataset: {cfg.dataset.path}...")
    dataset = load_dataset(
        path=cfg.dataset.path,
        name=cfg.dataset.name,
        split=cfg.dataset.split,
    )
    print("> Loaded dataset.\n")

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        # training duration and batch size
        per_device_train_batch_size=cfg.training.train_batch_size,
        max_steps=cfg.training.max_steps,
        # learning rate and scheduler (linear warmup + cosine decay)
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=cfg.training.warmup_steps,
        # optimizer (AdamW by default)
        weight_decay=0.1,
        adam_beta1=0.90,
        adam_beta2=0.95,
        adam_epsilon=1e-4,
        # regularization and training stability
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=1.0,
        # NOTE: mixed precision training, use bf16 if possible
        fp16=True,
        # NOTE: gradient checkpointing, trade speed for memory
        # gradient_checkpointing=True,
        # logging and saving
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        push_to_hub=True,
        hub_token=cfg.hf_token,
        # hub_private_repo=True,
        hub_strategy="all_checkpoints",
    )

    resume_path = None
    latest_checkpoint_folder = None
    if cfg.resume_from_checkpoint:
        repo_id = f"{username}/{cfg.training.output_dir}"
        repo_files = api.list_repo_files(repo_id=repo_id)

        checkpoint_numbers = []
        for file_path in repo_files:
            match = re.search(r"checkpoint-(\d+)", file_path)
            if match:
                checkpoint_numbers.append(int(match.group(1)))

        if not checkpoint_numbers:
            raise Exception("No checkpoints found.")

        latest_step = max(checkpoint_numbers)
        latest_checkpoint_folder = f"checkpoint-{latest_step}"
        print(f"Found latest checkpoint: {latest_checkpoint_folder}")

        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"*{latest_checkpoint_folder}/*"],
            local_dir=cfg.training.output_dir,
        )
        resume_path = f"{cfg.training.output_dir}/{latest_checkpoint_folder}"

    print("\n=== Training Configuration ===")
    if latest_checkpoint_folder:
        print(f"checkpoint: {latest_checkpoint_folder}")

    print(OmegaConf.to_yaml(cfg.training))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("> Beginning training...")

    trainer.train(resume_from_checkpoint=resume_path)

    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)

    print(f"> Training complete. Saved to {cfg.training.output_dir}")


if __name__ == "__main__":
    main()
