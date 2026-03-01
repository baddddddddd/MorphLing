import sentencepiece as spm
from transformers.models.llama import LlamaTokenizerFast
from tokenizers import SentencePieceBPETokenizer


class SentencePieceTokenizer(LlamaTokenizerFast):
    def __init__(
        self,
        tokenizer_file: str,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        **kwargs,
    ):
        super().__init__(
            tokenizer_file=tokenizer_file,
            unk_token=str(unk_token),
            bos_token=str(bos_token),
            eos_token=str(eos_token),
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )

        self.pad_token = self.unk_token

    @staticmethod
    def train(
        corpus_file: str,
        output_file: str,
        vocab_size: int,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        min_frequency: int = 2,
        **kwargs,
    ):
        def get_text_from_corpus():
            with open(corpus_file, "r") as f:
                lines = f.readlines()
                lines = list(map(lambda line: line.strip(), lines))
                batch_size = 1000
                for i in range(0, len(lines), batch_size):
                    batch = lines[i : i + batch_size]
                    yield batch

        tokenizer = SentencePieceBPETokenizer()
        tokenizer.train_from_iterator(
            get_text_from_corpus(),
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=True,
            special_tokens=[unk_token, bos_token, eos_token],
        )

        tokenizer.save(output_file)
