import math
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from ..tokenizers import MorphlingTokenizer, SentencePieceTokenizer, UnigramTokenizer

tokenizer_registry = {
    "MorphlingTokenizer": MorphlingTokenizer,
    "SentencePieceTokenizer": SentencePieceTokenizer,
    "UnigramTokenizer": UnigramTokenizer,
}


def calculate_root_preservation(tokenizer):
    module_root_dir = Path(__file__).parent.parent
    unimorph_filepath = module_root_dir / "resources" / "unimorph-tgl.txt"

    with open(unimorph_filepath, "r") as f:
        lines = f.readlines()

    fully_preserved_roots = 0
    partially_preserved_roots = 0
    total_words = 0

    SENTENCEPIECE_SPACE = "▁"
    for line in lines:
        root, *words, morphs = line.strip().split()
        for word in words:
            total_words += 1
            tokens = tokenizer.tokenize(word)
            tokens[0] = tokens[0].lstrip(SENTENCEPIECE_SPACE)
            if root in tokens:
                fully_preserved_roots += 1
                continue

            found_root = False
            for l in range(len(tokens)):
                if not root.startswith(tokens[l]):
                    continue

                for r in range(l, len(tokens)):
                    if not root.endswith(tokens[r]):
                        continue

                    attempt = "".join(tokens[l : r + 1])
                    if attempt == root:
                        found_root = True
                        break

                if found_root:
                    partially_preserved_roots += 1
                    break

    preserved_roots = fully_preserved_roots + partially_preserved_roots
    print(f"Total words: {total_words}")
    print(
        f"Fully preserved roots: {fully_preserved_roots} ({fully_preserved_roots / total_words * 100:.2f}%)"
    )
    print(
        f"Partially preserved roots: {partially_preserved_roots} ({partially_preserved_roots / total_words * 100:.2f}%)"
    )
    print(
        f"Total preserved roots: {preserved_roots} ({preserved_roots / total_words * 100:.2f}%)"
    )


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print("=== Active Configuration ===")
    print(OmegaConf.to_yaml(cfg))

    print(f"> Loading tokenizer: {cfg.tokenizer.name} @ {cfg.tokenizer.file}")
    TokenizerClass = tokenizer_registry[cfg.tokenizer.name]
    tokenizer = TokenizerClass(
        cfg.tokenizer.file,
        add_bos_token=False,
        add_eos_token=False,
    )

    print(f"> Successfully loaded tokenizer.\n")

    calculate_root_preservation(tokenizer)


if __name__ == "__main__":
    main()
