import os

import hydra
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
from itertools import chain
from omegaconf import DictConfig, OmegaConf

from ..tokenizers import MorphlingTokenizer, SentencePieceTokenizer


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

    if "repo_id" not in cfg:
        raise Exception(
            "repo_id is required, add +repo_id=REPO_ID_HERE when running command"
        )

    login(cfg.hf_token)

    print(f"\n> Loading dataset: {cfg.dataset.path}...")
    dataset = load_dataset(
        path=cfg.dataset.path,
        name=cfg.dataset.name,
        split=cfg.dataset.split,
    )
    print("> Loaded dataset.\n")

    def group_texts(examples):
        block_size = cfg.model.context_window

        concatenated_examples = {
            k: list(chain.from_iterable(examples[k])) for k in examples.keys()
        }

        total_length = len(next(iter(concatenated_examples.values())))

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        return result

    num_proc = os.cpu_count()
    if "num_proc" in cfg:
        num_proc = cfg.num_proc

    print("> Collating dataset...")
    dataset = dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    print(f"> Done processing dataset.")

    print(f"> Pushing to {cfg.repo_id}...")
    dataset = DatasetDict({"train": dataset})
    dataset.push_to_hub(cfg.repo_id)

    print(f"> Pushed collated dataset to {cfg.repo_id}.")


if __name__ == "__main__":
    main()
