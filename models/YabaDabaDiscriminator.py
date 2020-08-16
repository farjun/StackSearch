from models.DabaCnnAutoencoder import DabaCnnAutoencoder
import tensorflow as tf

def leaky_relu(*args, **kwargs):
    return tf.keras.layers.LeakyReLU(*args, **kwargs)

class DabaDiscriminator(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.down_d2 = tf.keras.layers.Dense(64, activation=leaky_relu())
        self.down_d3 = tf.keras.layers.Dense(32, activation=leaky_relu())
        self.densePredict = tf.keras.layers.Dense(1)

    def call(self, x, **kwargs):
        x = self.down_d2(x)
        x = self.down_d3(x)
        return self.densePredict(x)