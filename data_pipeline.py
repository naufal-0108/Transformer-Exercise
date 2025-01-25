import tensorflow as tf
import os, numpy as np
from tqdm.auto import tqdm
from tokenizers import Tokenizer

train_indo = r"C:\Users\naufal\Startup\datasets\train\train_text_indo.txt"
train_eng  = r"C:\Users\naufal\Startup\datasets\train\train_text_eng.txt"
val_indo = r"C:\Users\naufal\Startup\datasets\val\val_text_indo.txt"
val_eng  = r"C:\Users\naufal\Startup\datasets\val\val_text_eng.txt"

tokenizer_ind = Tokenizer.from_file(r"C:\Users\naufal\Startup\datasets\tokenizers\tokenizer_indo.json")
tokenizer_eng = Tokenizer.from_file(r"C:\Users\naufal\Startup\datasets\tokenizers\tokenizer_eng.json")

special_tokens = {"<start>", "<end>", "<pad>", "<unk>"}

with open(train_indo, 'r', encoding='utf-8') as file:
    train_content_indo = file.readlines()
    train_content_indo_clean  = [s.replace("\n", "").strip() for s in train_content_indo]

with open(train_eng, 'r', encoding='utf-8') as file:
    train_content_eng = file.readlines()
    train_content_eng_clean  = [s.replace("\n", "").strip() for s in train_content_eng]

with open(val_indo, 'r', encoding='utf-8') as file:
    val_content_indo = file.readlines()
    val_content_indo_clean  = [s.replace("\n", "").strip() for s in val_content_indo]

with open(val_eng, 'r', encoding='utf-8') as file:
    val_content_eng = file.readlines()
    val_content_eng_clean  = [s.replace("\n", "").strip() for s in val_content_eng]

train_indo_encoder_inputs = train_content_indo_clean
train_eng_decoder_inputs  = ["<start>" + s for s in tqdm(train_content_eng_clean)]
train_eng_decoder_targets = [s + "<end>" for s in tqdm(train_content_eng_clean)]

val_indo_encoder_inputs = val_content_indo_clean
val_eng_decoder_inputs = ["<start>" + s for s in tqdm(val_content_eng_clean)]
val_eng_decoder_targets = [s + "<end>" for s in tqdm(val_content_eng_clean)]


train_indo_encoded, train_eng_encoded_i, train_eng_encoded_t = [], [], []
val_indo_encoded, val_eng_encoded_i, val_eng_encoded_t = [], [], []

max_len = 128

for s in tqdm(train_indo_encoder_inputs):
    encoded_s = tokenizer_ind.encode(s).ids
    len_s = len(encoded_s)
    n_pad = max_len - len_s
    encoded_s = encoded_s + n_pad*[2]
    train_indo_encoded.append(encoded_s)

for s in tqdm(train_eng_decoder_inputs):
    encoded_s = tokenizer_eng.encode(s).ids
    len_s = len(encoded_s)
    n_pad = max_len - len_s
    encoded_s = encoded_s + n_pad*[2]
    train_eng_encoded_i.append(encoded_s)

for s in tqdm(train_eng_decoder_targets):
    encoded_s = tokenizer_eng.encode(s).ids
    len_s = len(encoded_s)
    n_pad = max_len - len_s
    encoded_s = encoded_s + n_pad*[2]
    train_eng_encoded_t.append(encoded_s)

for s in tqdm(val_indo_encoder_inputs):
    encoded_s = tokenizer_ind.encode(s).ids
    len_s = len(encoded_s)
    n_pad = max_len - len_s
    encoded_s = encoded_s + n_pad*[2]
    val_indo_encoded.append(encoded_s)

for s in tqdm(val_eng_decoder_inputs):
    encoded_s = tokenizer_eng.encode(s).ids
    len_s = len(encoded_s)
    n_pad = max_len - len_s
    encoded_s = encoded_s + n_pad*[2]
    val_eng_encoded_i.append(encoded_s)

for s in tqdm(val_eng_decoder_targets):
    encoded_s = tokenizer_eng.encode(s).ids
    len_s = len(encoded_s)
    n_pad = max_len - len_s
    encoded_s = encoded_s + n_pad*[2]
    val_eng_encoded_t.append(encoded_s)


def create_mask(tokens):
    mask = tf.cast(tf.equal(tokens, 2), dtype=tf.float32)
    return mask

print(train_indo_encoder_inputs[0])
print(train_eng_decoder_inputs[0])
print(train_eng_decoder_targets[0])
print()
print(train_indo_encoded[0])
print(train_eng_encoded_i[0])
print(train_eng_encoded_t[0])
print()
print(create_mask(train_indo_encoded[0]))

print("Saving files...", flush=True)
np.save(r"C:\Users\naufal\Startup\datasets\train\train_indo_encoded.npy", train_indo_encoded)
np.save(r"C:\Users\naufal\Startup\datasets\train\train_eng_encoded_i.npy", train_eng_encoded_i)
np.save(r"C:\Users\naufal\Startup\datasets\train\train_eng_encoded_t.npy", train_eng_encoded_t)
np.save(r"C:\Users\naufal\Startup\datasets\val\val_indo_encoded.npy", val_indo_encoded)
np.save(r"C:\Users\naufal\Startup\datasets\val\val_eng_encoded_i.npy", val_eng_encoded_i)
np.save(r"C:\Users\naufal\Startup\datasets\val\val_eng_encoded_t.npy", val_eng_encoded_t)


# print(train_indo_encoder_inputs[:3])
# print(train_eng_decoder_inputs[:3])
# print(train_eng_decoder_targets[:3])
# print()
# print(val_indo_encoder_inputs[:3])
# print(val_eng_decoder_inputs[:3])
# print(val_eng_decoder_targets[:3])

# max_train = dict()
# prev_ind, prev_eng = 0, 0
# max_val = dict()

# for ind, eng in zip(train_indo_encoded, train_eng_encoded_i):

#     len_ind = len(ind)
#     len_eng = len(eng)

#     if len_ind >= len_eng:
        
#         if len_ind > prev_ind:
#             max_train['ind'] = len_ind
#             prev_ind = len_ind

#         else:
#             continue
    
#     else:

#         if len_eng > prev_eng:
#             max_train['eng'] = len_eng
#             prev_eng = len_eng
            
#         else:
#             continue

# prev_ind, prev_eng = 0, 0

# for ind, eng in zip(val_indo_encoded, val_eng_encoded_i):

#     len_ind = len(ind)
#     len_eng = len(eng)

#     if len_ind >= len_eng:
        
#         if len_ind > prev_ind:
#             max_val['ind'] = len_ind
#             prev_ind = len_ind

#         else:
#             continue
    
#     else:

#         if len_eng > prev_eng:
#             max_val['eng'] = len_eng
#             prev_eng = len_eng
            
#         else:
#             continue




# print(max_train)
# print(max_val)


