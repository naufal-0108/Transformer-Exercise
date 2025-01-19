import os
from tokenizer import train_tokenizer

train_corpus_ind = "datasets/train/train_text_indo.txt"
train_corpus_eng = "datasets/train/train_text_eng.txt"
special_token = ["<start>", "<end>", "<pad>", "<unk>"]
save_dir_root = os.path.join("datasets", "tokenizers")
vocab_size = 50000

if __name__ == "__main__":
    print("Training tokenizer for Indonesian...")
    train_tokenizer([train_corpus_ind], vocab_size, os.path.join(save_dir_root, "tokenizer_indo.json"), special_token)
    print("Tokenizer for Indonesian trained.")
    print("Training tokenizer for English...")
    train_tokenizer([train_corpus_eng], vocab_size, os.path.join(save_dir_root, "tokenizer_eng.json"), special_token)
    print("Tokenizer for English trained.")