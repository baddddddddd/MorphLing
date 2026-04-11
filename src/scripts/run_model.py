import hydra
import torch
from huggingface_hub import login, HfApi
from omegaconf import DictConfig, OmegaConf
from transformers.tokenization_python import PreTrainedTokenizer
from transformers.models.auto import AutoModelForCausalLM

from ..tokenizers import MorphlingTokenizer, SentencePieceTokenizer, UnigramTokenizer


tokenizer_registry = {
    "MorphlingTokenizer": MorphlingTokenizer,
    "SentencePieceTokenizer": SentencePieceTokenizer,
    "UnigramTokenizer": UnigramTokenizer,
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

    api = HfApi()
    user = api.whoami()
    username = user["name"]

    model_id = f"{username}/{cfg.training.output_dir}"

    print(f"> Logged in as {username}")

    load_kwargs = {}
    if "checkpoint" in cfg:
        checkpoint_folder = f"checkpoint-{cfg.checkpoint}"
        load_kwargs["subfolder"] = checkpoint_folder
        print(f"> Loading {model_id} (subfolder: {checkpoint_folder})...")
    else:
        print(f"> Loading {model_id}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).to(device)
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id

    print(f"> Successfully loaded model with device: {device}\n")

    print(f"> Loading tokenizer: {cfg.tokenizer.name} @ {cfg.tokenizer.file}")
    TokenizerClass = tokenizer_registry[cfg.tokenizer.name]
    tokenizer = TokenizerClass(cfg.tokenizer.file)

    print(f"> Successfully loaded tokenizer.\n")

    EXIT_CMD = "/exit"
    print(f"Type {EXIT_CMD} to stop the inference loop.")

    while True:
        prompt = input("\n> ")
        if prompt == EXIT_CMD:
            break

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.model.context_window,
                temperature=0.8,
                top_p=0.90,
                do_sample=True,
            )
            ids = outputs[0]

        text = tokenizer.decode(ids, skip_special_tokens=True)
        print(text)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(f"=============\nTokens: {tokens}")


if __name__ == "__main__":
    main()
