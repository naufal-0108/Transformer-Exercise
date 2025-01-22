import os
from tokenizers import Tokenizer

tokenizer_indo = Tokenizer.from_file(r"C:\Users\naufal\Startup\datasets\tokenizers\tokenizer_indo.json")
tokenizer_eng = Tokenizer.from_file(r"C:\Users\naufal\Startup\datasets\tokenizers\tokenizer_eng.json")

if __name__ == "__main__":

    indo_text = "<start>saya suka makan nasi.<end><pad><pad><pad>"
    eng_text = "<start>i like eating rice.<end><pad><pad><pad>"

    indo_text_encoded = tokenizer_indo.encode(indo_text).ids
    eng_text_encoded = tokenizer_eng.encode(eng_text).ids

    print("Indonesian text encoded:", indo_text_encoded)
    print("English text encoded:", eng_text_encoded)
    print("Indonesian text decoded:", tokenizer_indo.decode(indo_text_encoded))
    print("English text decoded:", tokenizer_eng.decode(eng_text_encoded))
