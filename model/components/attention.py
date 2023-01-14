import tensorflow as tf
from keras.layers import Layer, Dense, Conv2D

def scaled_dot_product_attention(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: tf.Tensor = None):
    dk = tf.cast(k.shape[-1], dtype=tf.float32)

    attention_scores = tf.matmul(q, k, transpose_b=True)

    attention_scores = attention_scores/tf.sqrt(dk)

    if mask is not None:
        attention_scores += mask*(-1e20)

    attention_weights = tf.nn.softmax(attention_scores, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights

class MultiHeadAttention(Layer):
    def __init__(self, heads: int = 8, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.heading_sample = self.embedding_dim // self.heads

        self.linear_q = Dense(units=embedding_dim)
        self.linear_k = Dense(units=embedding_dim)
        self.linear_v = Dense(units=embedding_dim)

        self.linear_output = Dense(units=embedding_dim)

    def split(self, tensor: tf.Tensor):
        batch_size = tensor.shape[0]
        length = tensor.shape[1]

        tensor = tf.reshape(tensor, (batch_size, length, self.heads, self.heading_sample))
        tensor_heads = tf.transpose(tensor, [0, 2, 1, 3])

        return tensor_heads

    def call(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: tf.Tensor = None):
        batch_size = q.shape[0]
        length = q.shape[1]

        qw = self.linear_q(q)
        kw = self.linear_k(k)
        vw = self.linear_v(v)

        q_heads = self.split(qw)
        k_heads = self.split(kw)
        v_heads = self.split(vw)

        scaled_dot_output, attention_weights = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)

        tensor = tf.transpose(scaled_dot_output, [0, 2, 1, 3])
        tensor = tf.reshape(tensor, (batch_size, length, self.embedding_dim))

        output = self.linear_output(tensor)

        return output, attention_weights

class TwoDimAttention(Layer):
    def __init__(self, n: int, c: int, kernel_size: int | tuple, strides: int | tuple, padding: str):
        super().__init__()

        self.conv_q = Conv2D(filters=c, kernel_size=kernel_size, strides=strides, padding=padding)
        self.conv_k = Conv2D(filters=c, kernel_size=kernel_size, strides=strides, padding=padding)
        self.conv_v = Conv2D(filters=c, kernel_size=kernel_size, strides=strides, padding=padding)

        self.conv_output = Conv2D(filters=n, kernel_size=kernel_size, strides=strides, padding=padding)

    def call(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor):
        conv_q = self.conv_q(q) # dim = (batch_size, time, frequency, channel)
        conv_k = self.conv_k(k)
        conv_v = self.conv_v(v)

        q_freq = tf.transpose(conv_q, [0, 3, 1, 2])
        k_freq = tf.transpose(conv_k, [0, 3, 1, 2])
        v_freq = tf.transpose(conv_v, [0, 3, 1, 2])

        q_time = tf.transpose(conv_q, [0, 3, 2, 1])
        k_time = tf.transpose(conv_k, [0, 3, 2, 1])
        v_time = tf.transpose(conv_v, [0, 3, 2, 1])

        scaled_dot_freq, _ = scaled_dot_product_attention(q_freq, k_freq, v_freq)  # dim = (batch_size, channel, time, frequency)

        scaled_dot_time, _ = scaled_dot_product_attention(q_time, k_time, v_time)  # dim = (batch_size, channel, frequency, time)

        scaled_dot_freq = tf.transpose(scaled_dot_freq, [0, 2, 3, 1])
        scaled_dot_time = tf.transpose(scaled_dot_time, [0, 3, 2, 1])

        output = tf.concat([scaled_dot_time, scaled_dot_freq], axis=-1) # dim = (batch_size, time, frequency, 2*channel)

        output = self.conv_output(output)

        return output
  
