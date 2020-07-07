import tensorflow as tf
import typing


class BasicFCN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        self.input_dim = input_dim
        super().__init__(*args, **kwargs)
        self.down_d1 = tf.keras.layers.Dense(32, activation=leaky_relu())
        self.down_d2 = tf.keras.layers.Dense(16, activation=leaky_relu())
        self.down_d3 = tf.keras.layers.Dense(8, activation=leaky_relu())
        self.down_d4 = tf.keras.layers.Dense(output_dim, activation=leaky_relu())
        self.up_d3 = tf.keras.layers.Dense(8, activation=leaky_relu())
        self.up_d2 = tf.keras.layers.Dense(16, activation=leaky_relu())
        self.up_d1 = tf.keras.layers.Dense(32, activation=leaky_relu())
        self.out = tf.keras.layers.Dense(input_dim, activation=leaky_relu())

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.down_d1(x, training=training)
        x = self.down_d2(x, training=training)
        x = self.down_d3(x, training=training)
        x = self.down_d4(x, training=training)
        x = self.up_d3(x, training=training)
        x = self.up_d2(x, training=training)
        x = self.up_d1(x, training=training)
        x = self.out(x, training=training)
        return x

    def encode(self, inputs):
        x = inputs
        x = self.down_d1(x, training=False)
        x = self.down_d2(x, training=False)
        x = self.down_d3(x, training=False)
        x = self.down_d4(x, training=False)
        return x


def get_FCN_model(input_dim: int, output_dim: int) -> BasicFCN:
    return BasicFCN(input_dim, output_dim)


def leaky_relu(*args, **kwargs):
    return tf.keras.layers.LeakyReLU(*args, **kwargs)


if __name__ == '__main__':
    input_dim = 20
    model = get_FCN_model(input_dim, 10)
    input = tf.keras.layers.Input(shape=input_dim)
    model(input)
    model.summary()
