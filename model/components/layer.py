import tensorflow as tf
from keras.layers import Layer, LayerNormalization, Dropout
from model.components.attention import MultiHeadAttention
from model.components.ffn import PositionWiseFeedForwardNetwork
from typing import Union, Callable

class EncoderLayer(Layer):
    def __init__(self, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[tf.Tensor], tf.Tensor]]):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(epsilon=eps)
        self.multi_head_attention = MultiHeadAttention(heads=heads, embedding_dim=embedding_dim)
        self.layer_norm_2 = LayerNormalization(epsilon=eps)
        self.ffn = PositionWiseFeedForwardNetwork(d_ff=d_ff, embedding_dim=embedding_dim, activation=activation)
        
        self.dropout_1 = Dropout(rate=dropout_rate)
        self.dropout_2 = Dropout(rate=dropout_rate)

    def call(self, tensor: tf.Tensor, mask: tf.Tensor = None, is_train: bool = False):
        tensor_norm = self.layer_norm_1(tensor)
        multi_head_attention_output, _ = self.multi_head_attention(tensor_norm, tensor_norm, tensor_norm, mask)

        multi_head_attention_output = self.dropout_1(multi_head_attention_output, training=is_train)
        
        tensor = tensor + multi_head_attention_output

        tensor_norm = self.layer_norm_2(tensor)

        ffn_out = self.ffn(tensor_norm)
        ffn_out = self.dropout_2(ffn_out, training=is_train)

        tensor = tensor + ffn_out

        return tensor


class DecoderLayer(Layer):
    def __init__(self, embedding_dim: int, heads: int, d_ff: int, dropout_rate:float, eps: float, activation: Union[str, Callable[[tf.Tensor], tf.Tensor]]):
        super().__init__()
        self.layer_norm_1 = LayerNormalization(epsilon=eps)
        self.layer_norm_2 = LayerNormalization(epsilon=eps)
        self.layer_norm_3 = LayerNormalization(epsilon=eps)

        self.masked_multi_head_attention = MultiHeadAttention(heads=heads, embedding_dim=embedding_dim)
        self.multi_head_attention = MultiHeadAttention(heads=heads, embedding_dim=embedding_dim)
        self.ffn = PositionWiseFeedForwardNetwork(d_ff=d_ff, embedding_dim=embedding_dim, activation=activation)

        self.dropout_1 = Dropout(rate=dropout_rate)
        self.dropout_2 = Dropout(rate=dropout_rate)
        self.dropout_3 = Dropout(rate=dropout_rate)

    def call(self, tensor: tf.Tensor, encoder_output: tf.Tensor, look_ahead_mask: tf.Tensor = None, padding_mask: tf.Tensor = None, is_train: bool = False):
        tensor_norm = self.layer_norm_1(tensor)
        masked_multi_head_attention_output, _ = self.masked_multi_head_attention(tensor_norm, tensor_norm, tensor_norm, look_ahead_mask)
        masked_multi_head_attention_output = self.dropout_1(masked_multi_head_attention_output, training=is_train)
        tensor = tensor + masked_multi_head_attention_output

        tensor_norm = self.layer_norm_2(tensor)
        multi_head_attention_output, _ = self.multi_head_attention(tensor_norm, encoder_output, encoder_output, padding_mask)
        multi_head_attention_output = self.dropout_2(multi_head_attention_output, training=is_train)
        tensor = tensor + multi_head_attention_output

        tensor_norm = self.layer_norm_3(tensor)
        ffn_out = self.ffn(tensor_norm)
        ffn_out = self.dropout_3(ffn_out, training=is_train)
        tensor = tensor + ffn_out

        return tensor
