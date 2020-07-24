import tensorflow as tf
import typing


class BasicFCN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        self.input_dim = input_dim
        super().__init__(*args, **kwargs)
        reg_dict = {
            "kernel_regularizer": tf.keras.regularizers.l1_l2(),
            "bias_regularizer": tf.keras.regularizers.l1_l2(),
            "activity_regularizer": tf.keras.regularizers.l1_l2()
        }
        init_dict = {
            "kernel_initializer": tf.random_normal_initializer(),
            "bias_initializer": tf.random_normal_initializer()
        }
        common_dict = {**reg_dict, **init_dict}

        self.down_d1 = tf.keras.layers.Dense(128, activation=leaky_relu(), **common_dict)
        self.down_d2 = tf.keras.layers.Dense(64, activation=leaky_relu(), **common_dict)
        self.down_d3 = tf.keras.layers.Dense(32, activation=leaky_relu(), **common_dict)
        self.down_d4 = tf.keras.layers.Dense(output_dim, activation=tf.keras.activations.tanh, **common_dict)
        self.up_d3 = tf.keras.layers.Dense(32, activation=leaky_relu(), **common_dict)
        self.up_d2 = tf.keras.layers.Dense(64, activation=leaky_relu(), **common_dict)
        self.up_d1 = tf.keras.layers.Dense(128, activation=leaky_relu(), **common_dict)
        self.out = tf.keras.layers.Dense(input_dim, activation=leaky_relu(), **common_dict)

    def call(self, inputs, training=None, mask=None):
        return self.decode(self.encode(inputs, training), training)

    def encode(self, inputs, training=False):
        x = inputs
        x = self.down_d1(x, training=training)
        x = self.down_d2(x, training=training)
        x = self.down_d3(x, training=training)
        x = self.down_d4(x, training=training)
        return x

    def decode(self, inputs, training=False):
        x = inputs
        x = self.up_d3(x, training=training)
        x = self.up_d2(x, training=training)
        x = self.up_d1(x, training=training)
        x = self.out(x, training=training)
        return x


def get_FCN_model(input_dim: int, output_dim: int) -> BasicFCN:
    return BasicFCN(input_dim, output_dim)


def leaky_relu(*args, **kwargs):
    return tf.keras.layers.LeakyReLU(*args, **kwargs)


if __name__ == '__main__':
    input_dim = 20
    output_dim = 10
    model = get_FCN_model(input_dim, output_dim)
    input = tf.keras.layers.Input(shape=input_dim)
    model(input)
    model.summary()
