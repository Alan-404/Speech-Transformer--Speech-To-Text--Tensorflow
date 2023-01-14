import tensorflow as tf
from keras.layers import Layer, Conv2D, LayerNormalization
from model.components.pre_net import PreNet
from model.components.layer import EncoderLayer
from model.components.position import PostionalEncoding
from model.components.linear import Linear
from typing import Union, Callable

class Encoder(Layer):
    def __init__(self, n_layer: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation:  Union[str, Callable[[tf.Tensor], tf.Tensor]] ,m: int, n: int, c: int, kernel_size: int, strides: int | tuple, padding: str, max_length: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.conv_1 = Conv2D(filters=1, kernel_size=3, strides=2, padding='same')
        self.conv_2 = Conv2D(filters=1, kernel_size=3, strides=2, padding='same')

        self.pre_net = PreNet(m=m, n=n, c=c, kernel_size=kernel_size, strides=strides, padding=padding)
        self.linear = Linear(max_length=max_length, embedding_dim=embedding_dim)

        self.position = PostionalEncoding()
        self.conv_final = Conv2D(filters=1, kernel_size=kernel_size, strides=strides, padding=padding)

        self.encoder_layers = [EncoderLayer(embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, activation=activation, dropout_rate=dropout_rate, eps=eps) for _ in range(n_layer)]

        self.layer_norm = LayerNormalization(epsilon=eps)

    def call(self, tensor: tf.Tensor, is_train: bool):
        batch_size = tensor.shape[0]
        tensor = self.conv_1(tensor)
        tensor = self.conv_2(tensor)

        tensor = self.pre_net(tensor)
        tensor = self.conv_final(tensor)
        tensor = self.linear(tensor)
        
        tensor = tf.reshape(tensor, (batch_size, self.max_length, self.embedding_dim))

        tensor = tensor + self.position.encode_position(tensor.shape[1],self.embedding_dim)

        for layer in self.encoder_layers:
            tensor = layer(tensor, None, is_train)

        return tensor

        
        