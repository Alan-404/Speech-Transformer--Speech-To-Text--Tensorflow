import tensorflow as tf
from keras.layers import Layer, Dense

class Linear(Layer):
    def __init__(self, max_length: int, embedding_dim: int):
        super().__init__()
        self.linear_1 = Dense(units=embedding_dim)
        self.linear_2 = Dense(units=max_length)

    def call(self, tensor: tf.Tensor): 
        tensor = tf.transpose(tensor, [0, 1, 3, 2]) # dim = (batch_size, time, channel, frequency)
        tensor = self.linear_1(tensor)

        tensor = tf.transpose(tensor, [0, 3, 2, 1]) # dim = (batch_size, frequency, channel, time)
        tensor = self.linear_2(tensor)

        tensor = tf.transpose(tensor, [0, 3, 1, 2]) # dim = (batch_size, time, frequency, channel)

        return tensor