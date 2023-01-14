import tensorflow as tf
import numpy as np

def generate_padding_mask(tensor: tf.Tensor):
    return tf.cast(tensor == 0, dtype=tf.float32)[:, np.newaxis, np.newaxis, :]

def __generate_look_ahead_mask(inp_len):
    mask = 1 - tf.linalg.band_part(tf.ones((inp_len, inp_len)), -1, 0)
    return mask  # (length_seq, length_seq)

def generate_mask(tensor: tf.Tensor):

    padding_mask = generate_padding_mask(tensor)

    trig = __generate_look_ahead_mask(tensor.shape[1])

    look_ahead_mask = tf.math.maximum(trig, padding_mask)

    return padding_mask, look_ahead_mask