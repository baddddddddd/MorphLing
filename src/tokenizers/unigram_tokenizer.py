import os

from transformers.models.llama import LlamaTokenizerFast
from tokenizers import SentencePieceUnigramTokenizer


class UnigramTokenizer(LlamaTokenizerFast):
    def __init__(
        self,
        tokenizer_file: str,
        dataset=None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        vocab_size: int = 8000,
        **kwargs,
    ):
        if not os.path.exists(tokenizer_file):
            if dataset is None:
                raise Exception("dataset must be provided for corpus training")

            self._train(
                dataset=dataset,
                output_file=tokenizer_file,
                vocab_size=vocab_size,
                unk_token=str(unk_token),
                bos_token=str(bos_token),
                eos_token=str(eos_token),
            )

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

    def _train(
        self,
        dataset,
        output_file: str,
        vocab_size: int,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        **kwargs,
    ):
        def batch_iterator(batch_size=1000):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i : i + batch_size]["text"]

        tokenizer = SentencePieceUnigramTokenizer()
        tokenizer.train_from_iterator(
            batch_iterator(),
            vocab_size=vocab_size,
            show_progress=True,
            special_tokens=[unk_token, bos_token, eos_token],
        )

        tokenizer.save(output_file)
