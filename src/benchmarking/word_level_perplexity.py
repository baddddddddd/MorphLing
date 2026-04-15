import torch
import math

import hydra
import torchinfo
from huggingface_hub import login, HfApi
from omegaconf import DictConfig, OmegaConf
from transformers.models.llama import LlamaForCausalLM
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


def calculate_dataset_word_level_perplexity(
    dataset, model, tokenizer, text_column="text", device="cuda"
):
    model.eval()
    model.to(device)

    max_length = getattr(model.config, "max_position_embeddings", 2048)
    stride = max_length // 2

    total_dataset_nll = 0.0
    total_dataset_words = 0
    total_dataset_tokens = 0
    total_evaluated_tokens = 0
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
        total_dataset_tokens += total_tokens

        prev_end_loc = 0

        for begin_loc in range(0, total_tokens, stride):
            end_loc = min(begin_loc + max_length, total_tokens)
            trg_len = end_loc - prev_end_loc

            input_ids = input_ids_full[:, begin_loc:end_loc]
            target_ids = input_ids.clone()

            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                evaluated_tokens = trg_len - 1 if begin_loc == 0 else trg_len

                if evaluated_tokens > 0:
                    avg_nll_per_prediction = outputs.loss.item()
                    chunk_total_nll = avg_nll_per_prediction * evaluated_tokens

                    total_dataset_nll += chunk_total_nll
                    total_evaluated_tokens += evaluated_tokens
                    valid_chunks += 1

            prev_end_loc = end_loc
            if end_loc == total_tokens:
                break

    if total_dataset_words == 0:
        print("Error: No words found in the dataset.")
        return float("inf")

    dataset_word_normalized_nll = total_dataset_nll / total_dataset_words
    dataset_word_level_ppl = math.exp(dataset_word_normalized_nll)

    dataset_token_normalized_nll = total_dataset_nll / total_evaluated_tokens
    dataset_token_level_ppl = math.exp(dataset_token_normalized_nll)

    token_fertility_rate = total_dataset_tokens / total_dataset_words

    print("\n=== Dataset Evaluation Results ===")
    print(f"Total Chunks Evaluated: {valid_chunks}")
    print(f"Total Words: {total_dataset_words}")
    print(f"Total Sequence Tokens: {total_dataset_tokens}")
    print(f"Total Tokens Evaluated (Loss Targets): {total_evaluated_tokens}")
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
