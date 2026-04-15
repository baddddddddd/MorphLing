import math
import unicodedata
from pathlib import Path

import hydra
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


def normalize_to_ascii(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def calculate_root_preservation(tokenizer):
    module_root_dir = Path(__file__).parent.parent
    unimorph_filepath = module_root_dir / "resources" / "unimorph-tgl.txt"

    with open(unimorph_filepath, "r") as f:
        lines = f.readlines()

    exact_aligned_roots = 0
    over_segmented_roots = 0
    under_segmented_roots = 0
    total_words = 0

    SENTENCEPIECE_SPACE = "▁"
    seen = set()
    for line in lines:
        root, *words, morphs = line.strip().split()
        root = normalize_to_ascii(root).replace("'", "")

        for word in words:
            word = normalize_to_ascii(word).replace("'", "")
            if word in seen:
                continue

            seen.add(word)

            total_words += 1
            raw_tokens = tokenizer.tokenize(word)
            tokens = [t.replace(SENTENCEPIECE_SPACE, "") for t in raw_tokens]
            tokens = [t for t in tokens if t]

            # exact alignment
            if root in tokens:
                exact_aligned_roots += 1
                continue

            found_root = False

            # over-segmentation
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
                    over_segmented_roots += 1
                    break

            # under-segmentation
            if not found_root:
                for t in tokens:
                    if root in t:
                        under_segmented_roots += 1
                        found_root = True
                        break

            # misaligned
            if not found_root:
                print(f"Boundary Violation: {word} -> {root} -> {tokens}")

    recoverable_roots = (
        exact_aligned_roots + over_segmented_roots + under_segmented_roots
    )
    misaligned = total_words - recoverable_roots

    print(f"Total words evaluated: {total_words}")
    print(
        f"Exact Alignment: {exact_aligned_roots} ({exact_aligned_roots / total_words * 100:.2f}%)"
    )
    print(
        f"Over-segmentation: {over_segmented_roots} ({over_segmented_roots / total_words * 100:.2f}%)"
    )
    print(
        f"Under-segmentation: {under_segmented_roots} ({under_segmented_roots / total_words * 100:.2f}%)"
    )
    print(
        f"Total Recoverable Roots: {recoverable_roots} ({recoverable_roots / total_words * 100:.2f}%)"
    )
    print(
        f"Misaligned Segmentation: {misaligned} ({misaligned / total_words * 100:.2f}%)"
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
