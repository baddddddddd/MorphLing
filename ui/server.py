import os, sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, ROOT)

from src import MorphlingTokenizer, SentencePieceTokenizer

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))

print("Loading tokenizers…")
sp_tokenizer = SentencePieceTokenizer(
    os.path.join(ROOT, "data/tokenizer/sentencepiece-16k.json"),
    add_bos_token=False,
    add_eos_token=False,
)

ml_tokenizer = MorphlingTokenizer(
    os.path.join(ROOT, "data/tokenizer/morphling-16k.json"),
    add_bos_token=False,
    add_eos_token=False,
)
print("Tokenizers ready.")

MARKER = "\u241f"
SP_SPACE = "\u2581"  # ▁

MORPHEME_TAGS = [
    ("##PREFIX", "prefix"),
    ("##SUFFIX", "suffix"),
    ("##INFIX", "infix"),
    ("##REDUP", "redup"),
    ("##REPEAT", "repeat"),
    ("##CAPITAL", "capital"),
]


def classify_ml(token):
    if token in ("<s>", "</s>", "<unk>"):
        return "special"
    if token.endswith(MARKER):
        for tag, cat in MORPHEME_TAGS:
            if tag in token:
                return cat
        return "morpheme"
    if token.startswith(SP_SPACE):
        return "boundary"
    return "subword"


def display_ml(token):
    if not token.endswith(MARKER):
        # return token.replace(SP_SPACE, "\u00b7")
        return token.replace(SP_SPACE, "")
    for tag, _ in MORPHEME_TAGS:
        if tag in token:
            root = token[: token.index(tag)]
            label = tag.replace("##", "")
            return (root + "\u00a0[" + label + "]") if root else ("[" + label + "]")
    return token.replace(MARKER, "")


def display_sp(token):
    # return token.replace(SP_SPACE, "\u00b7")
    return token.replace(SP_SPACE, "")


def group_by_word(tokens, ids, is_ml=False):
    words, current = [], []
    for tok, tid in zip(tokens, ids):
        entry = {
            "text": tok,
            "display": display_ml(tok) if is_ml else display_sp(tok),
            "type": classify_ml(tok),
            "id": tid,
        }
        is_boundary = (
            tok in ("<s>", "</s>", "<unk>")
            or tok.startswith(SP_SPACE)
            or (len(tok) == 1 and not tok.isalnum() and not tok.endswith(MARKER))
        )
        if is_boundary and current:
            words.append({"tokens": current})
            current = []
        current.append(entry)
    if current:
        words.append({"tokens": current})
    return words


@app.route("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.route("/tokenize", methods=["POST"])
def tokenize():
    body = request.get_json()
    text = (body or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        sp_toks = sp_tokenizer.tokenize(text)
        ml_toks = ml_tokenizer.tokenize(text)

        sp_ids = list(sp_tokenizer.encode(text, add_special_tokens=False))
        ml_ids = list(ml_tokenizer.encode(text, add_special_tokens=False))

        sp_flat = [
            {
                "text": t,
                "display": display_sp(t),
                "type": "boundary" if t.startswith(SP_SPACE) else "subword",
                "id": i,
            }
            for t, i in zip(sp_toks, sp_ids)
        ]

        ml_flat = [
            {"text": t, "display": display_ml(t), "type": classify_ml(t), "id": i}
            for t, i in zip(ml_toks, ml_ids)
        ]

        char_len = len(text.replace(" ", ""))

        words_sp = group_by_word(sp_toks, sp_ids, is_ml=False)
        words_ml = group_by_word(ml_toks, ml_ids, is_ml=True)

        return jsonify(
            {
                "sentencepiece": {
                    "tokens": sp_flat,
                    "words": words_sp,
                    "count": len(sp_toks),
                    "tokens_per_word": round(len(sp_toks) / max(len(words_sp), 1), 2),
                },
                "morphling": {
                    "tokens": ml_flat,
                    "words": words_ml,
                    "count": len(ml_toks),
                    "tokens_per_word": round(len(ml_toks) / max(len(words_ml), 1), 2),
                },
                "char_len": char_len,
            }
        )

    except Exception as exc:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5000)
