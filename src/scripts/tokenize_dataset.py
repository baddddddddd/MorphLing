import os

import hydra
from datasets import load_dataset
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf

from ..tokenizers import MorphlingTokenizer, SentencePieceTokenizer


tokenizer_registry = {
    "MorphlingTokenizer": MorphlingTokenizer,
    "SentencePieceTokenizer": SentencePieceTokenizer,
}


def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count(),
    )

    return dataset


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print("=== Active Configuration ===")
    print(OmegaConf.to_yaml(cfg))

    if "hf_token" not in cfg:
        raise Exception(
            "hf_token is required, add +hf_token=YOUR_TOKEN_HERE when running command"
        )

    if "repo_id" not in cfg:
        raise Exception(
            "repo_id is required, add +repo_id=REPO_ID_HERE when running command"
        )

    login(cfg.hf_token)

    dataset = load_dataset(
        path=cfg.dataset.path,
        name=cfg.dataset.name,
        split=cfg.dataset.split,
    )

    TokenizerClass = tokenizer_registry[cfg.tokenizer.name]
    tokenizer = TokenizerClass(
        cfg.tokenizer.file,
        vocab_size=cfg.tokenizer.vocab_size,
        add_bos_token=True,
        add_eos_token=True,
    )

    print(f"Tokenizing dataset {cfg.dataset.path} with {cfg.tokenizer.name}")

    dataset = tokenize_dataset(dataset, tokenizer)
    print("Done tokenizing dataset.")

    print(f"Pushing to {cfg.repo_id}...")
    dataset.push_to_hub(cfg.repo_id, private=True)

    print(f"Pushed tokenized dataset to {cfg.repo_id}.")


if __name__ == "__main__":
    main()
