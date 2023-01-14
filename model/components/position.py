import tensorflow as tf
import numpy as np


class PostionalEncoding:
    def __create_encoded_length(self, length: int):
        pos = np.arange(length)
        pos = np.expand_dims(pos, axis=1)
        return pos

    def __create_encoded_embedded(self, embedding_dim: int):
        angles = np.arange(embedding_dim)
        angles[0::2] = angles[1::2]
        angles = 1/(np.power(10000, angles/embedding_dim))
        angles = np.expand_dims(angles, axis=0)

        return angles

    def encode_position(self, length: int, embedding_dim: int):
        pos = self.__create_encoded_length(length)
        
        angles = self.__create_encoded_embedded(embedding_dim)

        angles_pos = np.dot(pos, angles)
        angles_pos[0::2] = np.sin(angles_pos[0::2])
        angles_pos[1::2] = np.cos(angles_pos[1::2])

        angles_pos = np.expand_dims(angles_pos, axis=0)

        return tf.cast(angles_pos, dtype=tf.float32)