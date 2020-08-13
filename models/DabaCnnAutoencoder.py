import tensorflow as tf

class DabaCnnAutoencoder(tf.keras.Model):
    def __init__(self, input_dim, latent_space_dim, *args, **kwargs):
        self.input_dim = input_dim
        self.latent_space_dim = latent_space_dim
        super().__init__(*args, **kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.down_d1 = tf.keras.layers.Conv2D(300, kernel_size=(5,self.input_dim), strides=2, padding='same', activation='relu')
        self.down_d2 = tf.keras.layers.Conv2D(600, kernel_size=(5,1), strides=2)
        #todo self.down_d3 = should be tensorflow addon maxout layer (some addon layer we will try to add later)
        self.down_d4 = tf.keras.layers.Dense(latent_space_dim, activation=tf.keras.activations.tanh)
        self.up_d3 = tf.keras.layers.Dense(32, activation=leaky_relu())
        self.up_d2 = tf.keras.layers.Dense(64, activation=leaky_relu())
        self.up_d1 = tf.keras.layers.Dense(128, activation=leaky_relu())
        self.out = tf.keras.layers.Dense(input_dim, activation=leaky_relu())

    def call(self, inputs, training=None, mask=None):
        return self.decode(self.encode(inputs, training), training)

    def encode(self, inputs, training=False):
        x = inputs
        x = self.down_d1(x, training = training)
        x = self.down_d2(x, training = training)
        x = self.down_d4(x, training = training)
        return x

    def decode(self, inputs, training=False):
        x = inputs
        x = self.up_d3(x, training=training)
        x = self.up_d2(x, training=training)
        x = self.up_d1(x, training=training)
        x = self.out(x, training=training)
        return x

    def __str__(self):
        return "YabaDabaCnnAutoencoder_" + str(self.latent_space_dim)

def leaky_relu(*args, **kwargs):
    return tf.keras.layers.LeakyReLU(*args, **kwargs)


if __name__ == '__main__':
    input_dim = 20
    output_dim = 10
    model = DabaCnnAutoencoder(input_dim, output_dim)
    input = tf.keras.layers.Input(shape=input_dim)
    model(input)
    model.summary()
