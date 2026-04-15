import hydra
from huggingface_hub import login
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from tqdm import tqdm

from ..tokenizers import (
    MorphlingTokenizer,
    SentencePieceTokenizer,
    UnigramTokenizer,
    MorphlingTokenizerV2,
)

tokenizer_registry = {
    "MorphlingTokenizer": MorphlingTokenizer,
    "MorphlingTokenizerV2": MorphlingTokenizerV2,
    "SentencePieceTokenizer": SentencePieceTokenizer,
    "UnigramTokenizer": UnigramTokenizer,
}


def calculate_dataset_fertility_rate(dataset, tokenizer, text_column="text"):
    total_dataset_words = 0
    total_dataset_tokens = 0

    for item in tqdm(dataset, desc="Evaluating Dataset Fertility"):
        text = item[text_column]

        if not text or not text.strip():
            continue

        words = text.split()
        num_words = len(words)

        if num_words == 0:
            continue

        inputs = tokenizer.tokenize(text)
        if isinstance(inputs, dict) and "input_ids" in inputs:
            total_tokens = len(inputs["input_ids"])
        else:
            total_tokens = len(inputs)

        if total_tokens == 0:
            continue

        total_dataset_words += num_words
        total_dataset_tokens += total_tokens

    if total_dataset_words == 0:
        print("Error: No words found in the dataset.")
        return 0.0

    token_fertility_rate = total_dataset_tokens / total_dataset_words

    print("\n=== Dataset Evaluation Results ===")
    print(f"Total Words: {total_dataset_words}")
    print(f"Total Sequence Tokens: {total_dataset_tokens}")
    print(f"Token Fertility Rate: {token_fertility_rate:.4f}")

    return token_fertility_rate


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print("=== Active Configuration ===")
    print(OmegaConf.to_yaml(cfg))

    if "hf_token" in cfg:
        login(cfg.hf_token)

    TokenizerClass = tokenizer_registry[cfg.tokenizer.name]
    tokenizer = TokenizerClass(cfg.tokenizer.file)
    print(f"\n> Loading {cfg.tokenizer.name} with {cfg.tokenizer.file}")

    print(f"\n> Loading dataset: {cfg.dataset.path}...")
    dataset = load_dataset(
        path=cfg.dataset.path,
        name=cfg.dataset.name,
        split=cfg.dataset.split,
    )

    text_column_name = "text"

    calculate_dataset_fertility_rate(
        dataset=dataset,
        tokenizer=tokenizer,
        text_column=text_column_name,
    )


if __name__ == "__main__":
    main()
