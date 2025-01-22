import numpy as np
from PIL import Image
image = Image.open(r"C:\Users\asus\Downloads\Pas_Foto_Naufal.jpg").resize(size=(960, 1280))
image.save(r".\Pas_Foto_Naufal.jpg")
# import numpy as np
# import tensorflow as tf
# from tokenizers import Tokenizer
# from tensorflow.keras.layers import Embedding, Dense
# from torch import embedding


# def attention_mask(nd, ns, *, dtype):
#     """1's in the lower triangle, counting from the lower right corner.

#     Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
#     """
#     i = tf.range(nd)[:,None]
#     j = tf.range(ns)
#     m = i >= j - ns + nd
#     return tf.cast(m, dtype)

# sentence_inputs_ind = "<start>saya suka kamu.<pad><pad><pad>"
# sentence_inputs_eng = "<start>i love you.<pad><pad><pad>"
# sentence_target = "i like eating rice.<end><pad><pad><pad>"

# tokenizer_ind = Tokenizer.from_file(r"C:\Users\naufal\Startup\datasets\tokenizers\tokenizer_indo.json")
# tokenizer_eng = Tokenizer.from_file(r"C:\Users\naufal\Startup\datasets\tokenizers\tokenizer_eng.json")

# sentence_inputs_ind_encoded = np.array(tokenizer_ind.encode(sentence_inputs_ind).ids)
# sentence_inputs_eng_encoded = np.array(tokenizer_eng.encode(sentence_inputs_eng).ids)
# sentence_target_encoded = np.array(tokenizer_eng.encode(sentence_target).ids)

# sentence_inputs_ind_encoded = sentence_inputs_ind_encoded[None, :]
# sentence_inputs_eng_encoded = sentence_inputs_eng_encoded[None, :]
# sentence_target_encoded = sentence_target_encoded[None, :]

# embedding_layer = Embedding(50000, 8)
# q = Dense(8, activation='linear')
# k = Dense(8, activation='linear')

# q_out = q(embedding_layer(sentence_inputs_ind_encoded))
# k_out = k(embedding_layer(sentence_inputs_ind_encoded))

# pad_mask = 1 - tf.cast(tf.math.equal(sentence_inputs_ind_encoded, 2), tf.float32)
# pad_mask = pad_mask[:, tf.newaxis, :]
# causal_mask = attention_mask(q_out.shape[1], k_out.shape[1], dtype=q_out.dtype)
# causal_mask = causal_mask[tf.newaxis, :, :]

# merge_mask = causal_mask * pad_mask
# print(pad_mask)
# print(causal_mask)
# print(merge_mask)

# w = tf.matmul(q_out, k_out, transpose_b=True)
# # pad_mask = tf.cast(tf.math.equal(sentence_inputs_ind_encoded, 2), tf.float32)
# # causal_mask = tf.linalg.set_diag(tf.linalg.band_part(tf.ones((q_out.shape[1], q_out.shape[1])), 0, -1), tf.zeros(q_out.shape[1]))
# # merge_mask = tf.cast(tf.maximum(causal_mask, pad_mask), dtype=tf.boll)
# # merge_mask = tf.Variable(merge_mask, trainable=False)
# w_masked = w*merge_mask - tf.cast(1e10, w.dtype)*(1-merge_mask)
# w_softmax = tf.nn.softmax(w_masked, axis=-1)
# # print(pad_mask)
# # print(causal_mask)
# # print(merge_mask)
# print(w)
# print(w_masked)
# print(w_softmax)
# # print(w_softmax)
# # w_masked = w + (pad_mask * -1e9)
# # w_softmax = tf.nn.softmax(w_masked, axis=-1)

# # print("Input ind:")
# # print(sentence_inputs_ind_encoded)
# # print(len(sentence_inputs_ind_encoded))
# # print("w:")
# # print(w)
# # print("w_masked:")
# # print(w_masked)
# # print("w_softmax:")
# # print(w_softmax)

# # tensor_x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
# # tensor_triu = tf.linalg.band_part(tensor_x, 0, -1)
# # tensor_triu_2 = tf.linalg.set_diag(tensor_triu, tf.zeros(tensor_x.shape[0], dtype=tf.float32))
# # print(tensor_triu)
# # print(tensor_triu_2)
# # print(tensor_x)