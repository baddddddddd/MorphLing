from src import MorphlingTokenizer, SentencePieceTokenizer

if __name__ == "__main__":
    s = 'Tumitigil-tigil kayo, at kung hindi ay "pagbabalibagin" ko \'yong lahat.'
    # s = "Ako'y may lobo.\nLumipad nga lang sa langit tangina"
    # s = "ABCDE"
    # s = "Jose P. Rizal"
    # s = "Ang mga prutas ng Pilipinas ay kilala sa buong mundo dahil sa kanilang matamis na lasa at natatanging amoy. Ang mangga (Mangifera indica) ay itinuturing na pambansang prutas ng Pilipinas. Ang Philippine Carabao mango ay kilala bilang isa sa pinakamataas na kalidad na mangga sa buong mundo. Ito ay matamis, malambot, at mabango na may napakagandang kulay na ginto."

    tokenizer = MorphlingTokenizer("test2", corpus_file="dataset.txt")
    # SentencePieceTokenizer.train("dataset.txt", "test", 1000)
    # tokenizer = SentencePieceTokenizer("test")

    tokens = tokenizer.tokenize(s)
    token_ids = tokenizer.encode(s)
    dec = tokenizer.decode(token_ids[::])
    back = tokenizer.convert_tokens_to_string(tokens[::])
    print(s)
    print()
    print(tokens)
    print()
    print(token_ids)
    print()
    print(back)
    print()
    print(dec)
    print()
