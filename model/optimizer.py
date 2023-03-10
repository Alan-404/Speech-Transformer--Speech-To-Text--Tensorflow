import tensorflow as tf
from keras.optimizers.schedules.learning_rate_schedule import LearningRateSchedule

class CustomLearningRate(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps = 4000):
        super(CustomLearningRate, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        
    def __call__(self, step_num):
        lrate = tf.cast(self.d_model, tf.float32) ** (-0.5) * tf.math.minimum(step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5) )
        return lrate