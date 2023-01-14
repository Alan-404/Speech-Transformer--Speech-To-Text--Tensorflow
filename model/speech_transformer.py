import tensorflow as tf
from keras.models import Model
from model.utils.mask import generate_mask, generate_padding_mask
from keras.activations import relu
from model.components.encoder import Encoder
from model.components.decoder import Decoder
from typing import Union, Callable
from model.optimizer import CustomLearningRate
from keras.losses import SparseCategoricalCrossentropy
import os

class __SpeechTransformer(Model):
    def __init__(self, 
                vocab_size: int,
                max_length: int,
                n_layer: int=6, 
                embedding_dim: int=512, 
                heads: int=8, 
                d_ff: int=2048, 
                dropout_rate: float=0.1, 
                eps: float=0.1, 
                activation: Union[str, Callable[[tf.Tensor], tf.Tensor]] = relu, 
                m: int = 2, n: int = 64, c: int = 64, 
                kernel_size: int | tuple = 3, 
                strides: int | tuple = 3, 
                padding: str = "same"):
        super().__init__()
        self.encoder = Encoder(n_layer=n_layer, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation, m=m, n=n, c=c, kernel_size=kernel_size, strides=strides, padding=padding, max_length=max_length, embedding_dim=embedding_dim)
        self.decoder = Decoder(vocab_size=vocab_size, n_layer=n_layer, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation)

    def call(self, inp: tf.Tensor, targ: tf.Tensor, is_train: bool):
        targ_padding_mask, targ_look_ahead_mask = generate_mask(targ)

        encoder_output = self.encoder(inp, is_train)
        decoder_output = self.decoder(targ, encoder_output, targ_look_ahead_mask,targ_padding_mask, is_train)

        return decoder_output


class SpeechTransformer:
    def __init__(self,
                vocab_size: int,
                max_length: int,
                n_layer: int=6, 
                embedding_dim: int=512, 
                heads: int=8, 
                d_ff: int=2048, 
                dropout_rate: float=0.1, 
                eps: float=0.1, 
                activation: Union[str, Callable[[tf.Tensor], tf.Tensor]] = relu, 
                m: int = 2, n: int = 64, c: int = 64, 
                kernel_size: int | tuple = 3, 
                strides: int | tuple = 3, 
                padding: str = "same"):
        self.model = __SpeechTransformer(vocab_size, max_length, n_layer, embedding_dim, heads, d_ff, dropout_rate, eps, activation, m, n, c, kernel_size, strides, padding)
        self.lrate = CustomLearningRate(d_model=embedding_dim)


    def cal_acc(self, real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(real == 0)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        return tf.math.reduce_sum(accuracies) / tf.math.reduce_sum(mask)

    def loss_function(self, real, pred):        
        cross_entropy = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        
        mask = tf.math.logical_not(real == 0)

        loss = cross_entropy(real, pred)
        
        mask = tf.cast(mask, dtype=loss.dtype)
        
        loss = loss*mask
        
        return tf.math.reduce_sum(loss) / tf.math.reduce_sum(mask)
    
    def train_step(self, inp, targ):
    
        with tf.GradientTape() as tape:
            preds = self.model(inp, targ, True)

            d_loss = self.loss_function(targ, preds)

        grads = tape.gradient(d_loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss.update_state(d_loss)
        self.train_accuracy.update_state(self.cal_acc(targ, preds))
    
    def fit(self, data, epochs = 1, saved_checkpoint_at = 1):
        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        
        for epoch in range(epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (batch, (inp, targ)) in enumerate(data):
                self.train_step(inp, targ)

                if batch % 50 == 0:
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.3f} Accuracy {self.train_accuracy.result():.3f}')
                if (epoch + 1) % saved_checkpoint_at == 0:
                    saved_path = self.checkpoint_manager.save()
                    print('Checkpoint was saved at {}'.format(saved_path))
