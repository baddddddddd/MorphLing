import torch
import math

import hydra
import torchinfo
from huggingface_hub import login, HfApi
from omegaconf import DictConfig, OmegaConf
from transformers.models.llama import LlamaForCausalLM
from datasets import load_dataset
from tqdm import tqdm

from ..tokenizers import MorphlingTokenizer, SentencePieceTokenizer

tokenizer_registry = {
    "MorphlingTokenizer": MorphlingTokenizer,
    "SentencePieceTokenizer": SentencePieceTokenizer,
}


def calculate_dataset_word_level_perplexity(
    dataset, model, tokenizer, text_column="text", device="cuda"
):
    model.eval()
    model.to(device)

    max_length = getattr(model.config, "max_position_embeddings", 2048)

    total_dataset_nll = 0.0
    total_dataset_words = 0
    total_dataset_tokens = 0
    valid_chunks = 0

    for item in tqdm(dataset, desc="Evaluating Dataset"):
        text = item[text_column]

        if not text or not text.strip():
            continue

        words = text.split()
        num_words = len(words)

        if num_words == 0:
            continue

        inputs = tokenizer(text, return_tensors="pt")
        input_ids_full = inputs["input_ids"].to(device)
        total_tokens = input_ids_full.size(1)

        if total_tokens < 2:
            continue

        total_dataset_words += num_words

        for i in range(0, total_tokens, max_length):
            input_ids = input_ids_full[:, i : i + max_length]
            num_tokens = input_ids.size(1)

            if num_tokens < 2:
                continue

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                avg_nll_per_prediction = outputs.loss.item()
                chunk_total_nll = avg_nll_per_prediction * (num_tokens - 1)

            total_dataset_nll += chunk_total_nll
            total_dataset_tokens += num_tokens
            valid_chunks += 1

    if total_dataset_words == 0:
        print("Error: No words found in the dataset.")
        return float("inf")

    dataset_word_normalized_nll = total_dataset_nll / total_dataset_words
    dataset_word_level_ppl = math.exp(dataset_word_normalized_nll)

    dataset_token_normalized_nll = total_dataset_nll / (
        total_dataset_tokens - valid_chunks
    )
    dataset_token_level_ppl = math.exp(dataset_token_normalized_nll)

    token_fertility_rate = total_dataset_tokens / total_dataset_words

    print("\n=== Dataset Evaluation Results ===")
    print(f"Total Chunks Evaluated: {valid_chunks}")
    print(f"Total Words: {total_dataset_words}")
    print(f"Total Tokens: {total_dataset_tokens}")
    print(f"Token Fertility Rate: {token_fertility_rate:.2f}")
    print(f"Dataset Token-Level Perplexity: {dataset_token_level_ppl:.2f}")
    print(f"Dataset Word-Level Perplexity: {dataset_word_level_ppl:.2f}")

    return dataset_word_level_ppl


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
    print(f"\n> Loading {cfg.tokenizer.name} with {cfg.tokenizer.file}")

    model_id = f"{username}/{cfg.training.output_dir}"

    load_kwargs = {}
    if "checkpoint" in cfg:
        checkpoint_folder = f"checkpoint-{cfg.checkpoint}"
        load_kwargs["subfolder"] = checkpoint_folder
        print(f"\n> Loading {model_id} (subfolder: {checkpoint_folder})")
    else:
        print(f"\n> Loading {model_id}")

    model = LlamaForCausalLM.from_pretrained(model_id, **load_kwargs)

    print()
    torchinfo.summary(model)

    print(f"\n> Loading dataset: {cfg.dataset.path}...")
    dataset = load_dataset(
        path=cfg.dataset.path,
        name=cfg.dataset.name,
        split=cfg.dataset.split,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n> Moving model to {device} and starting evaluation...")

    text_column_name = "text"

    calculate_dataset_word_level_perplexity(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        text_column=text_column_name,
        device=device,
    )


if __name__ == "__main__":
    main()
