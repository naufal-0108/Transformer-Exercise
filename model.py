import os, math
import numpy as np
from pyparsing import C
from sympy import shape
import tensorflow as tf
from tokenizers import Tokenizer
from tensorflow.keras.layers import Embedding, Dense
from torch import dtype, mode

tf.random.set_seed(42)

class EmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embedding_size, initializer=None, stddev=0.01, mean=0.0):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.stddev = stddev
        self.mean = mean
        self.initializer = initializer
        if self.initializer is None:
            self.initializer = tf.random_normal_initializer(mean=self.mean,
                                                            stddev=self.stddev)

    def build(self, input_shape):

        with tf.name_scope("embedding_weights"):
            self.embedding_weights = self.add_weight(
                name="weights",
                shape=(self.vocab_size, self.embedding_size),
                dtype="float32",
                initializer=self.initializer
            )
        super(EmbeddingLayer, self).build(input_shape)

    def call(self, inputs, mode="embedding", scale=False):
        if mode == "embedding":
            return self.embedding(inputs, scale=scale)
        elif mode == "projection":
            return self.projection(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def embedding(self, inputs, scale=False):
        with tf.name_scope("embedding"):
            # Create binary mask of size [batch_size, length]
            inputs = tf.cast(inputs, tf.int32)
            embeddings = tf.nn.embedding_lookup(self.embedding_weights, inputs)

            # Scale embedding by the sqrt of the hidden size
            if scale:
                embeddings *= self.embedding_size ** 0.5

            return embeddings

    def projection(self, inputs):
        with tf.name_scope("output_layer"):
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]

            h_flat = tf.reshape(inputs, [-1, self.embedding_size])
            logits = tf.matmul(h_flat, self.embedding_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, seq_len, self.vocab_size])

class Conv1D(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, weights_init_stdev=0.02, weights_mean=0.0, bias_init=0.0):
        super(Conv1D, self).__init__()

        self.weights_init_stdev = weights_init_stdev
        self.weights_mean = weights_mean
        self.bias_init = bias_init
        self.in_channels = in_channels
        self.out_channels = out_channels

    def build(self, input_shape):
        self.weight = self.add_weight(shape=[self.in_channels, self.out_channels],
                                      dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(stddev=self.weights_init_stdev, mean=self.weights_mean))
        self.bias = self.add_weight(shape=[self.out_channels], initializer=tf.constant_initializer(self.bias_init))
        super(Conv1D, self).build(input_shape)

    def call(self, inputs):
        out_shape = [tf.shape(inputs)[0], tf.shape(inputs)[1]] + [self.out_channels]
        inputs = tf.reshape(inputs, [-1, self.in_channels])
        matmul_out = tf.matmul(inputs, self.weight) + self.bias
        return tf.reshape(matmul_out, out_shape)
    
class FeedForward(tf.keras.layers.Layer):

    def __init__(self, hidden_size, filter_size, activation=tf.nn.relu):
        super(FeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.activation = activation
        self.dense_layer = Conv1D(self.hidden_size, self.filter_size)
        self.output_dense_layer = Conv1D(self.filter_size, self.hidden_size)

    def call(self, x):
        output = self.dense_layer(x)
        output = self.activation(output)
        output = self.output_dense_layer(output)
        return output


class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=1024):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.positional_encoding = self._create_positional_encoding()

    def _create_positional_encoding(self):
        # Create a positional encoding matrix of size (max_len, d_model)
        positions = np.arange(self.max_len)[:, np.newaxis]  # Shape: (max_len, 1)
        dimensions = np.arange(self.d_model)[np.newaxis, :]  # Shape: (1, d_model)

        # Calculate the angle rates
        angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / self.d_model)

        # Apply sine to even indices in the array and cosine to odd indices
        pos_encoding = np.zeros((self.max_len, self.d_model))
        pos_encoding[:, 0::2] = np.sin(positions * angle_rates[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(positions * angle_rates[:, 0::2])

        return tf.constant(pos_encoding, dtype=tf.float32)

    def call(self, x):
        # Add positional encoding to the input tensor
        seq_len = tf.shape(x)[1]
        return x + self.positional_encoding[:seq_len, :]

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = d_model // num_heads
        self.d_model = d_model
        self.q = Conv1D(d_model, d_model)
        self.k = Conv1D(d_model, d_model)
        self.v = Conv1D(d_model, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, pad_mask=None):
        x_shape = tf.shape(x)
        pad_mask = pad_mask[:, tf.newaxis, tf.newaxis, :]

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q_reshape = tf.reshape(q, (x_shape[0], x_shape[1], self.num_heads, self.dim_head))
        k_reshape = tf.reshape(k, (x_shape[0], x_shape[1], self.num_heads, self.dim_head))
        v_reshape = tf.reshape(v, (x_shape[0], x_shape[1], self.num_heads, self.dim_head))

        q_transposed = tf.transpose(q_reshape, perm=[0, 2, 1, 3])
        k_transposed = tf.transpose(k_reshape, perm=[0, 2, 1, 3])
        v_transposed = tf.transpose(v_reshape, perm=[0, 2, 1, 3])

        w = tf.matmul(q_transposed, k_transposed, transpose_b=True)

        if pad_mask is not None:
            w += (pad_mask * -1e9)

        w_norm = w / tf.math.sqrt(tf.cast(self.dim_head, tf.float32))
        w_softmax = tf.nn.softmax(w_norm, axis=-1)
        w_softmax = self.dropout(w_softmax)
        w_out = tf.reshape(tf.matmul(w_softmax, v_transposed),(x_shape))

        return self.layer_norm(w_out + x)
    
class MultiHeadCrossAttention(tf.keras.layers.Layer):
    
    def __init__(self, q_dim, k_dim, num_heads, dropout_rate=0.1):
        
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = q_dim // num_heads

        self.q = Conv1D(q_dim, q_dim)
        self.k = Conv1D(k_dim, k_dim)
        self.v = Conv1D(k_dim, k_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, q, k, pad_mask=None):

        q_shape = tf.shape(q)
        k_shape = tf.shape(k)

        pad_mask = pad_mask[:, tf.newaxis, tf.newaxis, :]

        q, k, v = self.q(q), self.k(k), self.v(k)

        q_reshape = tf.reshape(q, (q_shape[0], q_shape[1], self.num_heads, self.dim_head))
        k_reshape = tf.reshape(k, (k_shape[0], k_shape[1], self.num_heads, self.dim_head))
        v_reshape = tf.reshape(v, (k_shape[0], k_shape[1], self.num_heads, self.dim_head))

        q_transposed = tf.transpose(q_reshape, perm=[0, 2, 1, 3])
        k_transposed = tf.transpose(k_reshape, perm=[0, 2, 1, 3])
        v_transposed = tf.transpose(v_reshape, perm=[0, 2, 1, 3])

        w = tf.matmul(q_transposed, k_transposed, transpose_b=True)

        if pad_mask is not None:
            w += (pad_mask * -1e9)

        w_norm = w / tf.math.sqrt(tf.cast(self.dim_head, tf.float32))
        w_softmax = tf.nn.softmax(w_norm, axis=-1)
        w_softmax = self.dropout(w_softmax)
        w_out = tf.reshape(tf.matmul(w_softmax, v_transposed), q_shape)
        return self.layer_norm(w_out + q)
    
class MultiHeadCausalAttention(tf.keras.layers.Layer):

    # @tf.function
    def causal_mask(self, nd, ns, *, dtype):
        """1's in the lower triangle, counting from the lower right corner.

        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:,None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        assert d_model % num_heads == 0
        super(MultiHeadCausalAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = d_model // num_heads
        self.q = Conv1D(d_model, d_model)
        self.k = Conv1D(d_model, d_model)
        self.v = Conv1D(d_model, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, pad_mask=None):
        x_shape = tf.shape(x)
        causal_mask = self.causal_mask(x_shape[1], x_shape[1], dtype=x.dtype)
        pad_mask = pad_mask[:, tf.newaxis, tf.newaxis, :]
        pad_mask = 1 - pad_mask
        merge_mask = causal_mask * pad_mask
        q, k, v = self.q(x), self.k(x), self.v(x)
        q_reshape, k_reshape, v_reshape = [tf.reshape(x, (x_shape[0],  x_shape[1], self.num_heads, self.dim_head)) for x in [q, k, v]]
        q_transposed, k_transposed, v_transposed = [tf.transpose(x, perm=[0, 2, 1, 3]) for x in [q_reshape, k_reshape, v_reshape]]

        w = tf.matmul(q_transposed, k_transposed, transpose_b=True)

        if pad_mask is not None:
            w = w*merge_mask - tf.cast(1e10, dtype=w.dtype)*(1-merge_mask)
        
        w_norm = w / tf.math.sqrt(tf.cast(self.dim_head, tf.float32))
        w_softmax = tf.nn.softmax(w_norm, axis=-1)
        w_softmax = self.dropout(w_softmax)
        w_out = tf.reshape(tf.matmul(w_softmax, v_transposed), x_shape)
        return self.layer_norm(w_out + x)


    
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, n_dense=512, dropout_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.mlp = FeedForward(d_model, n_dense, activation=tf.nn.gelu)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x, mask):

        assert x.shape[1] == mask.shape[1]

        att_outs = self.mha(x, mask)
        mlp_outs = self.mlp(att_outs)
        mlp_outs = self.dropout(mlp_outs)

        return self.layer_norm(mlp_outs + att_outs)
    
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, q_dim, k_dim, num_heads, n_dense=512, dropout_rate=0.1):
        assert q_dim % num_heads == 0 and k_dim % num_heads == 0
        super(DecoderBlock, self).__init__()
        self.causal_att = MultiHeadCausalAttention(q_dim, num_heads)
        self.cross_att = MultiHeadCrossAttention(q_dim, k_dim, num_heads)
        self.mlp = FeedForward(q_dim, n_dense, activation=tf.nn.gelu)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, q, k, q_pad_mask, k_pad_mask):

        assert q.shape[1] == q_pad_mask.shape[1]
        assert k.shape[1] == k_pad_mask.shape[1]

        causal_att_outs = self.causal_att(q, q_pad_mask)
        cross_att_outs = self.cross_att(causal_att_outs, k, k_pad_mask)
        mlp_outs = self.mlp(cross_att_outs)
        mlp_outs = self.dropout(mlp_outs)

        return self.layer_norm(mlp_outs + cross_att_outs)
    

class Block(tf.keras.layers.Layer):

    def __init__(self, q_dim, k_dim, num_heads, n_dense, dropout_rate=0.1):
        super(Block, self).__init__()
        self.encoder = EncoderBlock(q_dim, num_heads, n_dense=n_dense, dropout_rate=dropout_rate)
        self.decoder = DecoderBlock(q_dim, k_dim, num_heads, n_dense=n_dense, dropout_rate=dropout_rate)

    def call(self, x_encoder, x_decoder, mask_encoder, mask_decoder):
        encoder_outs = self.encoder(x_encoder, mask_encoder)
        decoder_outs = self.decoder(x_decoder, encoder_outs, mask_decoder, mask_encoder)
        return encoder_outs, decoder_outs
    


class Transformer(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, n_dense, num_classes, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.blocks = [Block(d_model, d_model, num_heads, n_dense=n_dense, dropout_rate=dropout_rate) for _ in range(num_layers)]
    
    def call(self, x_encoder, x_decoder, mask_encoder, mask_decoder):

        for block in self.blocks:
            x_encoder, x_decoder = block(x_encoder, x_decoder, mask_encoder, mask_decoder)

        return x_encoder

def transfomer_model(num_layers, d_model, num_heads, n_dense, num_vocabs_ind, num_vocabs_eng, max_len_ind, max_len_eng, dropout_rate=0.1):

    inputs_encoder = tf.keras.Input(shape=(max_len_ind,), dtype=tf.int32)
    inputs_decoder = tf.keras.Input(shape=(max_len_eng,), dtype=tf.int32)
    inputs_encoder_mask = tf.keras.Input(shape=(max_len_ind,), dtype=tf.float32)
    inputs_decoder_mask = tf.keras.Input(shape=(max_len_eng,), dtype=tf.float32)

    embedding_layer_ind = EmbeddingLayer(num_vocabs_ind, d_model)
    embedding_layer_eng = EmbeddingLayer(num_vocabs_eng, d_model)

    pos_encoding_ind = SinusoidalPositionalEncoding(d_model, max_len=max_len_ind)
    pos_encoding_eng = SinusoidalPositionalEncoding(d_model, max_len=max_len_eng)

    x_encoder =  pos_encoding_ind(embedding_layer_ind(inputs_decoder))
    x_decoder =  pos_encoding_eng(embedding_layer_eng(inputs_decoder))

    transformer = Transformer(num_layers, d_model, num_heads, n_dense, num_classes=num_vocabs_eng, dropout_rate=dropout_rate)
    outputs = transformer(x_encoder, x_decoder, inputs_encoder_mask, inputs_decoder_mask)
    outputs = embedding_layer_eng(outputs, mode="projection")

    model = tf.keras.Model(inputs=[inputs_encoder, inputs_decoder, inputs_encoder_mask, inputs_decoder_mask], outputs=outputs)

    return model

sentence_inputs_ind = "saya suka makan nasi.<pad><pad><pad>"
sentence_inputs_eng = "<start>i like eating rice.<pad><pad><pad>"
sentence_target = "i like eating rice.<end><pad><pad><pad>"

tokenizer_ind = Tokenizer.from_file(r"C:\Users\naufal\Startup\datasets\tokenizers\tokenizer_indo.json")
tokenizer_eng = Tokenizer.from_file(r"C:\Users\naufal\Startup\datasets\tokenizers\tokenizer_eng.json")

sentence_inputs_ind_encoded = np.array(tokenizer_ind.encode(sentence_inputs_ind).ids)
sentence_inputs_eng_encoded = np.array(tokenizer_eng.encode(sentence_inputs_eng).ids)
sentence_target_encoded = np.array(tokenizer_eng.encode(sentence_target).ids)

sentence_inputs_ind_encoded = sentence_inputs_ind_encoded[None, :]
sentence_inputs_eng_encoded = sentence_inputs_eng_encoded[None, :]
sentence_target_encoded = sentence_target_encoded[None, :]

print(f"Input Indo: {sentence_inputs_ind_encoded}")
print(f"Input Eng: {sentence_inputs_eng_encoded}")
print(f"Target: {sentence_target_encoded}")
print()

num_layers = 8
d_model = 768
num_heads = 8
n_dense = 512
num_vocabs_ind = 8192
num_vocabs_eng = 8192
max_len_ind = 512
max_len_eng = 512
dropout_rate = 0.1

transformers = transfomer_model(num_layers=num_layers,
                                d_model=d_model,
                                num_heads=num_heads,
                                n_dense=n_dense,
                                num_vocabs_ind=num_vocabs_ind,
                                num_vocabs_eng=num_vocabs_eng,
                                max_len_ind=max_len_ind,
                                max_len_eng=max_len_eng,
                                dropout_rate=dropout_rate)

# embedding_layer = EmbeddingLayer(vocab_size=50000, embedding_size=8)
# # print(embedding_layer.embedding_weights.shape)

# embedding_out = embedding_layer(sentence_inputs_eng_encoded)
# projection_out = embedding_layer(embedding_out, mode="projection")

# print(embedding_layer.embedding_weights.shape)

# print(embedding_out)
# print(projection_out.shape)

print(transformers.summary())
