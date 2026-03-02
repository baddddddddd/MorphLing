import math

import hydra
import torchinfo
from omegaconf import DictConfig, OmegaConf
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

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

    TokenizerClass = tokenizer_registry[cfg.model.tokenizer]
    tokenizer = TokenizerClass(cfg.tokenizer_file)

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


if __name__ == "__main__":
    main()
