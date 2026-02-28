import re

from nltk import word_tokenize

from tglstemmer import stemmer


class MorphlingTokenizer:
    def __init__(self):
        self.SPECIAL_TOKEN_MARKER = "\u241f"

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
        self.PUNCTUATION_CHARS = self.PUNCTS_SPACE_AFTER | self.PUNCTS_SPACE_BEFORE

        self.VOWEL_CHARS = set("aeiou")

        self.VALID_CONTRACTIONS = set(["'y", "'t"])

        self.EOS_TOKEN = "<s>"
        self.BOS_TOKEN = "</s>"
        self.UNK_TOKEN = "<unk>"

    def _tokenize_word(self, word: str) -> list:
        # TODO: normalize single quotes to double quotes if context is quoting and not contractions

        # TODO: capitalization check, not robust but fast
        is_capital = word[0].isupper() and (len(word) == 1 or word[-1].islower())

        stem = stemmer.get_stem(word)
        root = str(stem)
        tokens = [root]

        if stem.dup:
            tokens.append(self.REPEAT_TAG)

        # TODO: handle single letter partial reduplication, e.g., aalis -> a + alis
        if stem.rep:
            tokens.append(self.REDUP_TAG)

        if stem.inf:
            tokens.append(stem.inf + self.INFIX_TAG)

        # TODO: handle phoneme change and assimilation
        if stem.pre:
            tokens.append(stem.pre + self.PREFIX_TAG)

        # TODO: handle phoneme change
        # vowel loss and metathesis doesn't change meaning so its ok?
        if stem.suf:
            tokens.append(stem.suf + self.SUFFIX_TAG)

        if stem.contraction:
            tokens.append(stem.contraction + self.SUFFIX_TAG)

        if is_capital:
            tokens.append(self.CAPITAL_TAG)

        # print(stem.__dict__)

        return tokens

    def _split_to_words(self, s: str) -> list:
        # words = word_tokenize(s)
        words = re.findall(r"[\w']+(?:-\w+)*|[^\w\s]|\n", s, re.UNICODE)
        return words

    def tokenize(self, s: str) -> list:
        words = self._split_to_words(s)
        tokens = []

        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens += word_tokens

        return tokens

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

        stem = word_tokens[0]
        for i in range(1, len(word_tokens)):
            token = word_tokens[i]
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

    def detokenize(self, tokens: list) -> str:
        word_token_buf = []
        words = []

        # TODO: EOS tokens should be used here instead
        tokens.append("tapos")

        for token in tokens:
            if not self._is_special_token(token):
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
            else:
                if no_space_next:
                    concat.append(word)
                    no_space_next = False
                else:
                    concat.append(" " + word)

        return "".join(concat).strip()
