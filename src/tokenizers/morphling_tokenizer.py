import os
import re
import string
from pathlib import Path

from datasets import concatenate_datasets, Dataset
from tglstemmer import stemmer
from tokenizers import SentencePieceBPETokenizer
from transformers.tokenization_python import PreTrainedTokenizer

from .sentencepiece_tokenizer import SentencePieceTokenizer


class MorphlingTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        bpe_tokenizer_file: str,
        dataset=None,
        vocab: str | dict | list | None = None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        add_bos_token: bool = True,
        add_eos_token: bool = False,
        vocab_size: int = 8000,
        min_frequency: int = 2,
        **kwargs,
    ):
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        module_root_dir = Path(__file__).parent.parent

        self.PREFIXES_FILE = module_root_dir / "resources" / "affixes" / "prefixes.txt"
        self.SUFFIXES_FILE = module_root_dir / "resources" / "affixes" / "suffixes.txt"
        self.INFIXES_FILE = module_root_dir / "resources" / "affixes" / "infixes.txt"
        self.WORDLIST_FILE = module_root_dir / "resources" / "tgl_wordlist.txt"

        # NOTE: MAKE SURE TO UPDATE THIS AS YOU ADD MORE SPECIAL TOKENS
        self.SPECIAL_TOKEN_COUNT = 126

        self._load_wordlist()

        # keep newlines and split on space
        # keep punctuations
        # handle words like araw-araw, 'Yon, and di'ba
        self.WORD_SPLIT_PATTERN = r"[\w']+(?:-\w+)*|[^\w\s]|\n"
        self.word_split_regex = re.compile(self.WORD_SPLIT_PATTERN, re.UNICODE)

        # train on corpus_file if tokenizer_file doesn't exist yet
        if not os.path.exists(bpe_tokenizer_file):
            if dataset is None:
                raise Exception("dataset must be provided for corpus training")

            self._train_bpe(
                dataset=dataset,
                output_file=bpe_tokenizer_file,
                vocab_size=vocab_size - self.SPECIAL_TOKEN_COUNT,
                unk_token=str(unk_token),
                bos_token=str(bos_token),
                eos_token=str(eos_token),
                min_frequency=min_frequency,
            )

        self.bpe_tokenizer = SentencePieceTokenizer(
            tokenizer_file=bpe_tokenizer_file,
            unk_token=str(unk_token),
            bos_token=str(bos_token),
            eos_token=str(eos_token),
            add_bos_token=False,
            add_eos_token=False,
        )

        # for O(1) identification if token is special
        self.SPECIAL_TOKEN_MARKER = "\u241f"
        self.SENTENCEPIECE_SPACE = "▁"

        self.PREFIX_TAG = "##PREFIX" + self.SPECIAL_TOKEN_MARKER
        self.SUFFIX_TAG = "##SUFFIX" + self.SPECIAL_TOKEN_MARKER
        self.INFIX_TAG = "##INFIX" + self.SPECIAL_TOKEN_MARKER
        self.REDUP_TAG = "##REDUP" + self.SPECIAL_TOKEN_MARKER
        self.REPEAT_TAG = "##REPEAT" + self.SPECIAL_TOKEN_MARKER
        self.CAPITAL_TAG = "##CAPITAL" + self.SPECIAL_TOKEN_MARKER

        self.PREFIX_TAG_LEN = len(self.PREFIX_TAG)
        self.SUFFIX_TAG_LEN = len(self.SUFFIX_TAG)
        self.INFIX_TAG_LEN = len(self.INFIX_TAG)
        self.REDUP_TAG_LEN = len(self.REDUP_TAG)
        self.REPEAT_TAG_LEN = len(self.REPEAT_TAG)
        self.CAPITAL_TAG_LEN = len(self.CAPITAL_TAG)

        self.PUNCTS_SPACE_AFTER = set(".,?!:;)]\"'")
        self.PUNCTS_SPACE_BEFORE = set("([")
        self.PUNCTS_NO_SPACE = set("\n\t")
        self.PUNCTUATION_CHARS = (
            self.PUNCTS_SPACE_AFTER | self.PUNCTS_SPACE_BEFORE | self.PUNCTS_NO_SPACE
        )

        self.VOWEL_CHARS = set("aeiou")

        self.VALID_CONTRACTIONS = set(["'y", "'t"])

        self.vocab = vocab
        if self.vocab is None:
            self._setup_vocab(
                unk_token=str(unk_token),
                bos_token=str(bos_token),
                eos_token=str(eos_token),
            )

        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        super().__init__(
            unk_token=str(unk_token),
            bos_token=str(bos_token),
            eos_token=str(eos_token),
            **kwargs,
        )

        self.pad_token = self.unk_token

        self.SEQUENCE_TOKENS = set(
            [
                self.bos_token,
                self.eos_token,
                self.unk_token,
            ]
        )

        self.memo = {}

    def _load_wordlist(self):
        self.wordlist = set()
        with open(self.WORDLIST_FILE, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    self.wordlist.add(line)

    def _setup_vocab(
        self,
        unk_token,
        bos_token,
        eos_token,
    ):
        self.vocab = dict(self.bpe_tokenizer.get_vocab())

        # prefixes
        with open(self.PREFIXES_FILE, "r") as f:
            lines = f.readlines()
            for line in lines:
                prefix = line.strip()
                prefix_token = prefix + self.PREFIX_TAG
                self.vocab[prefix_token] = len(self.vocab)

        # suffixes
        with open(self.SUFFIXES_FILE, "r") as f:
            lines = f.readlines()
            for line in lines:
                suffix = line.strip()
                suffix_token = suffix + self.SUFFIX_TAG
                self.vocab[suffix_token] = len(self.vocab)

        # infixes
        with open(self.INFIXES_FILE, "r") as f:
            lines = f.readlines()
            for line in lines:
                infix = line.strip()
                infix_token = infix + self.INFIX_TAG
                self.vocab[infix_token] = len(self.vocab)

        # full redup
        self.vocab[self.REPEAT_TAG] = len(self.vocab)

        # partial redup
        self.vocab[self.REDUP_TAG] = len(self.vocab)

        # capital
        self.vocab[self.CAPITAL_TAG] = len(self.vocab)

    def _tokenize(self, text: str) -> list:
        words = self._split_to_words(text)
        tokens = []

        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens += word_tokens

        bos_token = [self.bos_token] if self.add_bos_token else []
        eos_token = [self.eos_token] if self.add_eos_token else []

        return bos_token + tokens + eos_token

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        word_token_buf = []
        words = []

        # just a trick coz lazy to write if statements
        tokens.append(self.SENTENCEPIECE_SPACE)

        for token in tokens:
            # if not self._is_special_token(token):
            if self._is_word_boundary(token):
                word = self._detokenize_word(word_token_buf)
                if word:
                    words.append(word)
                word_token_buf.clear()

            word_token_buf.append(token)

        concat = []
        no_space_next = False
        opened_double_quotes = False
        for word in words:
            if len(word) == 1 and word in self.PUNCTUATION_CHARS:
                if word == '"':
                    if opened_double_quotes:
                        concat.append(word)
                    else:
                        concat.append(" " + word)
                        no_space_next = True

                    opened_double_quotes = not opened_double_quotes

                elif word in self.PUNCTS_SPACE_AFTER:
                    concat.append(word)
                elif word in self.PUNCTS_SPACE_BEFORE:
                    concat.append(" " + word)
                    no_space_next = True
                elif word in self.PUNCTS_NO_SPACE:
                    concat.append(word)
                    no_space_next = True

            else:
                if no_space_next:
                    concat.append(word)
                    no_space_next = False
                else:
                    concat.append(" " + word)

        return "".join(concat).strip()

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def get_vocab(self):
        return self.vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize_word(self, word: str) -> list:
        if len(word) == 1:
            return [word]

        # NOTE: capitalization check, not robust but fast
        is_capital = word[0].isupper() and word[-1].islower()

        # memoize because stemming is expensive
        word_key = word.lower()
        if word_key in self.memo:
            stem = self.memo[word_key]
        else:
            stem = stemmer.get_stem(word)

        root = str(stem)
        # print(stem.__dict__)

        special_tokens = []
        if root in self.wordlist:
            self.memo.setdefault(word_key, stem)

            if stem.dup:
                special_tokens.append(self.REPEAT_TAG)

            if stem.rep:
                special_tokens.append(self.REDUP_TAG)

            if stem.inf:
                special_tokens.append(stem.inf + self.INFIX_TAG)

            if stem.pre:
                special_tokens.append(stem.pre + self.PREFIX_TAG)

            # NOTE: phoneme change, assimilation, vowel loss, and metathesis doesn't change meaning so its ok for now
            if stem.suf:
                special_tokens.append(stem.suf + self.SUFFIX_TAG)

            if stem.contraction:
                special_tokens.append(stem.contraction + self.SUFFIX_TAG)

            if is_capital:
                special_tokens.append(self.CAPITAL_TAG)
        else:
            # either a proper noun or non-tagalog word
            root = word

        # TODO: perform SentencePiece BPE on root word
        bpe_tokens = self.bpe_tokenizer.tokenize(root)
        tokens = bpe_tokens + special_tokens
        return tokens

    def _split_to_words(self, text: str) -> list:
        words = self.word_split_regex.findall(text)
        return words

    def _reconstruct_full_reduplication(self, stem: str) -> str:
        new_stem = f"{stem}-{stem}"
        return new_stem

    def _reconstruct_partial_reduplication(self, stem: str) -> str:
        if stem[0] in self.VOWEL_CHARS:
            v = stem[0]
            new_stem = v + stem
        else:
            cv = stem[:2]
            new_stem = cv + stem

        return new_stem

    def _reconstruct_infix(self, stem: str, infix_token: str) -> str:
        infix = infix_token[:2]
        new_stem = stem[0] + infix + stem[1:]
        return new_stem

    def _get_suffix(self, suffix_token: str) -> str:
        suffix_stem = suffix_token[: -self.SUFFIX_TAG_LEN]
        return suffix_stem

    def _reconstruct_suffix(self, stem: str, suffix_token: str) -> str:
        suffix = self._get_suffix(suffix_token)
        new_stem = stem + suffix
        return new_stem

    def _get_prefix(self, prefix_token: str) -> str:
        prefix_stem = prefix_token[: -self.PREFIX_TAG_LEN]
        return prefix_stem

    def _reconstruct_prefix(self, stem: str, prefix_token: str) -> str:
        prefix = self._get_prefix(prefix_token)
        new_stem = prefix + stem
        return new_stem

    def _reconstruct_capitalization(self, stem: str) -> str:
        new_stem = stem.capitalize()
        return new_stem

    def _detokenize_word(self, word_tokens: list) -> str:
        if not word_tokens:
            return ""

        # TODO: SORT SPECIAL TOKENS FIRST THEN RECONSTRUCT SEQUENTIALLY
        # full redup -> partial redup -> infix -> suffix -> prefix

        # recover original root word fragmented by BPE
        root_tokens = []
        i = 0
        while i < len(word_tokens):
            token = word_tokens[i]
            if self._is_special_token(token):
                break

            root_tokens.append(token)
            i += 1

        stem = "".join(root_tokens).lstrip(self.SENTENCEPIECE_SPACE)

        if stem in self.SEQUENCE_TOKENS:
            return stem

        while i < len(word_tokens):
            token = word_tokens[i]
            i += 1
            if token.endswith(self.REPEAT_TAG):
                stem = self._reconstruct_full_reduplication(stem)
                continue

            if token.endswith(self.REDUP_TAG):
                stem = self._reconstruct_partial_reduplication(stem)
                continue

            if token.endswith(self.INFIX_TAG):
                stem = self._reconstruct_infix(stem, token)
                continue

            if token.endswith(self.SUFFIX_TAG):
                stem = self._reconstruct_suffix(stem, token)
                continue

            if token.endswith(self.PREFIX_TAG):
                stem = self._reconstruct_prefix(stem, token)
                continue

            if token.endswith(self.CAPITAL_TAG):
                stem = self._reconstruct_capitalization(stem)
                continue

        return stem

    def _is_special_token(self, token: str) -> bool:
        if not token:
            return False

        return token[-1] == self.SPECIAL_TOKEN_MARKER

    def _is_word_boundary(self, token: str) -> bool:
        if not token or self._is_special_token(token):
            return False

        if token in self.PUNCTUATION_CHARS:
            return True

        return token[0] == self.SENTENCEPIECE_SPACE

    def _preprocess(self, example):
        line = example["text"].strip()
        if not line:
            return {"text": ""}

        words = self._split_to_words(line)

        tokens = []
        for word in words:
            if len(word) <= 1:
                continue

            stem = stemmer.get_stem(word)
            if stem in self.wordlist:
                tokens.append(stem)
            else:
                tokens.append(word)

        processed = " ".join(tokens)
        return {"text": processed}

    def _train_bpe(
        self,
        dataset,
        output_file: str,
        vocab_size: int,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        min_frequency: int = 2,
        **kwargs,
    ):
        dataset = dataset.filter(
            lambda x: x["text"] != "",
            num_proc=os.cpu_count(),
        )

        dataset = dataset.map(
            lambda example: self._preprocess(example),
            remove_columns=dataset.column_names,
            num_proc=os.cpu_count(),
        )

        char_dataset = Dataset.from_dict(
            {"text": ["".join(chr(i) for i in range(256))]}
        )
        dataset = concatenate_datasets([char_dataset, dataset])

        def batch_iterator(batch_size=1000):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i : i + batch_size]["text"]

        tokenizer = SentencePieceBPETokenizer()

        tokenizer.train_from_iterator(
            batch_iterator(),
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=True,
            special_tokens=[unk_token, bos_token, eos_token],
        )

        tokenizer.save(output_file)
