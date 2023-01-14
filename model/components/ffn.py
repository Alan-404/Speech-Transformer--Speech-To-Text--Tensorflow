import tensorflow as tf
from keras.layers import Layer, Dense
from keras.activations import relu
from typing import Union, Callable

class PositionWiseFeedForwardNetwork(Layer):
    def __init__(self, d_ff: int, embedding_dim: int, activation: str | Union[str, Callable[[tf.Tensor], tf.Tensor]] = relu):
        super().__init__()
        self.hidden_layer = Dense(units=d_ff)
        self.activation = activation
        self.output_layer = Dense(units=embedding_dim)

    def call(self, tensor: tf.Tensor):
        tensor = self.hidden_layer(tensor)
        tensor = self.activation(tensor)
        tensor = self.output_layer(tensor)
        return tensor