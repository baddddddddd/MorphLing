import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
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


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print("=== Active Configuration ===")
    print(OmegaConf.to_yaml(cfg))

    dataset = load_dataset(
        path=cfg.dataset.path,
        name=cfg.dataset.name,
        split=cfg.dataset.split,
    )

    TokenizerClass = tokenizer_registry[cfg.tokenizer.name]
    tokenizer = TokenizerClass(
        cfg.tokenizer.file,
        dataset=dataset,
        vocab_size=cfg.tokenizer.vocab_size,
    )

    print("Done training tokenizer.")


if __name__ == "__main__":
    main()
