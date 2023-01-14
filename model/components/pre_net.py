import tensorflow as tf
from keras.layers import Layer
from model.components.attention import TwoDimAttention

class PreNet(Layer):
    def __init__(self, m: int, n: int, c: int, kernel_size: int | tuple, strides: int | tuple, padding: str):
        super().__init__()

        self.attention_layers = [TwoDimAttention(n, c, kernel_size, strides, padding) for _ in range(m)]

    def call(self, tensor: tf.Tensor):
        for layer in self.attention_layers:
            tensor = layer(tensor, tensor, tensor)
        return tensor