import tensorflow as tf

class DataBuilder:
    def build_dataset(self, x_train: tf.Tensor, y_train: tf.Tensor, batch_size: int, buffer_size: int, num_data: int | str = "all"):
        dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x_train, dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.int64)))
        dataset = dataset.shuffle(buffer_size).batch(batch_size)

        return dataset
        