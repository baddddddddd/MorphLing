import math

import hydra
import torchinfo
from datasets import load_dataset
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf
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

    TokenizerClass = tokenizer_registry[cfg.tokenizer.name]
    tokenizer = TokenizerClass(cfg.tokenizer.file)

    print("=== LLaMa Configuration ===")
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

    dataset = load_dataset(
        path=cfg.dataset.path,
        name=cfg.dataset.name,
        split=cfg.dataset.split,
    )

    args = TrainingArguments(
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
        hub_private_repo=True,
        hub_strategy="all_checkpoints",
    )

    print("\n=== Training Configuration ===")
    print(OmegaConf.to_yaml(cfg.training))


if __name__ == "__main__":
    main()
