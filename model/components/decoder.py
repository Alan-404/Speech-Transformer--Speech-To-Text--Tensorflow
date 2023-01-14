import tensorflow as tf
from keras.layers import Layer, LayerNormalization, Embedding, Dense
from model.components.position import PostionalEncoding
from model.components.layer import DecoderLayer
from typing import Union, Callable

class Decoder(Layer):
    def __init__(self, vocab_size: int, n_layer: int, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[tf.Tensor], tf.Tensor]]):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.position = PostionalEncoding()

        self.decoder_layers = [DecoderLayer(embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n_layer)]

        self.layer_norm = LayerNormalization(epsilon=eps)
        self.linear = Dense(units=vocab_size)

    def call(self, tensor: tf.Tensor, encoder_output: tf.Tensor, look_ahead_mask: tf.Tensor, padding_mask: tf.Tensor, is_train: bool):
        tensor = self.embedding_layer(tensor)
        tensor = tensor + self.position.encode_position(tensor.shape[1], self.embedding_dim)
        
        for layer in self.decoder_layers:
            tensor = layer(tensor,encoder_output, look_ahead_mask, padding_mask, is_train)

        tensor = self.layer_norm(tensor)
        output = self.linear(tensor)

        return output



