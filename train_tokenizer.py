import os, argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

def train_tokenizer(text_files, vocab_size, save_dir, special_tokens):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    tokenizer.train(files=text_files, trainer=trainer)
    tokenizer.save(save_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train tokenizer for Indonesian and English")
    parser.add_argument("--train_corpus_ind", type=str, help="Path to the training corpus for Indonesian")
    parser.add_argument("--train_corpus_eng", type=str, help="Path to the training corpus for English")
    parser.add_argument("--vocab_size", type=int, help="Size of the vocabulary")
    parser.add_argument("--save_dir_root", type=str, help="Path to the directory to save the tokenizers")
    parser.add_argument("--special_tokens", type=str, nargs="+", help="Special tokens to add to the tokenizer")

    args_dict = parser.parse_args().__dict__
    print("training tokenizer for Indonesian...", flush=True)
    train_tokenizer([args_dict["train_corpus_ind"]], args_dict["vocab_size"], os.path.join(args_dict["save_dir_root"], "tokenizer_indo.json"), args_dict["special_tokens"])
    print("tokenizer for Indonesian trained.", flush=True)
    print("training tokenizer for English...", flush=True)
    train_tokenizer([args_dict["train_corpus_eng"]], args_dict["vocab_size"], os.path.join(args_dict["save_dir_root"], "tokenizer_eng.json"), args_dict["special_tokens"])
    print("tokenizer for English trained.", flush=True)

