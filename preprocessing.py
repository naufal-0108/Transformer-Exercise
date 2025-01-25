import os
from datasets import load_dataset

print("Loading dataset...")
ds_train = load_dataset("cobrayyxx/FLEURS_INDO-ENG_Speech_Translation", split="train", num_proc=2)
ds_val = load_dataset("cobrayyxx/FLEURS_INDO-ENG_Speech_Translation", split="validation")
print("Dataset loaded.")
print("Creating text indo files...")

SAVE_DIR_PATH = "datasets"

os.makedirs(os.path.join(SAVE_DIR_PATH, "train"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR_PATH, "val"), exist_ok=True)

with open(os.path.join(SAVE_DIR_PATH,"train","train_text_indo.txt"), "w", encoding='utf-8') as f:
    for i in range(len(ds_train)):
        f.write(ds_train[i]["text_indo"].lower().strip() + "\n")

with open(os.path.join(SAVE_DIR_PATH,"val","val_text_indo.txt"), "w", encoding='utf-8') as f:
    for i in range(len(ds_val)):
        f.write(ds_val[i]["text_indo"].lower().strip() + "\n")

print("Text indo files created.")
print("Creating text eng files...")

with open(os.path.join(SAVE_DIR_PATH,"train","train_text_eng.txt"), "w", encoding='utf-8') as f:
    for i in range(len(ds_train)):
        f.write(ds_train[i]["text_en"].lower().strip() + "\n")

with open(os.path.join(SAVE_DIR_PATH,"val","val_text_eng.txt"), "w",encoding='utf-8') as f:
    for i in range(len(ds_val)):
        f.write(ds_val[i]["text_en"].lower().strip() + "\n")

print("Text eng files created.")
