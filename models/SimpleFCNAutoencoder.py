import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2DTranspose, Dense, Reshape, BatchNormalization

from hparams import HParams


def get_reg(use_reg: bool):
    l1 = HParams.REGULARIZER_L1
    l2 = HParams.REGULARIZER_L2
    return {} if not use_reg else {
        "kernel_regularizer": tf.keras.regularizers.l1_l2(l1, l2),
        "bias_regularizer": tf.keras.regularizers.l1_l2(l1, l2)
    }


class SimpleFCNAutoencoder(tf.keras.Model):
    def __init__(self, featureDim, latent_space_dim, useNormalization=False, *args, **kwargs):
        self.useNormalization = useNormalization
        self.inputshape = (HParams.MAX_SENTENCE_DIM, featureDim)
        self.latent_space_dim = latent_space_dim
        super().__init__(*args, **kwargs)

        if self.useNormalization:
            self.batchNormalization1 = BatchNormalization()
            self.batchNormalization2 = BatchNormalization()
            self.batchNormalization3 = BatchNormalization()

        # encode
        self.flatten = tf.keras.layers.Flatten()
        self.down_d1 = tf.keras.layers.Dense(100, activation='relu', **get_reg(HParams.USE_REGULARIZER))
        self.down_d2 = tf.keras.layers.Dense(200, activation="relu", **get_reg(HParams.USE_REGULARIZER))
        self.down_d3 = tf.keras.layers.Dense(latent_space_dim, activation="sigmoid", **get_reg(HParams.USE_REGULARIZER))

        # decode
        self.up_d1 = Dense(512, activation='relu', **get_reg(HParams.USE_REGULARIZER))
        self.up_d2 = Dense(self.inputshape[0] * self.inputshape[1] , activation='relu', **get_reg(HParams.USE_REGULARIZER))
        self.reshaper = Reshape((int(self.inputshape[0]), int(self.inputshape[1])))

    def call(self, inputs, training=None, mask=None):
        return self.decode(self.encode(inputs, training), training)

    def encode(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.down_d1(x, training=training)
        x = self.batchNormalization1(x, training=training) if self.useNormalization else x
        x = self.down_d2(x, training=training)
        x = self.batchNormalization2(x, training=training) if self.useNormalization else x
        x = self.down_d3(x, training=training)
        return x

    def decode(self, inputs, training=False):
        x = inputs
        x = self.up_d1(x, training=training)
        x = self.batchNormalization3(x, training=training) if self.useNormalization else x
        x = self.up_d2(x, training=training)
        x = self.reshaper(x)
        return x

    def __str__(self):
        return "SimpleFCNAutoencoder_" + str(self.latent_space_dim)


if __name__ == '__main__':
    model = SimpleCnnAutoencoder(HParams.getFeatureExtractorDim(), HParams.OUTPUT_DIM)
    input = tf.keras.layers.Input(shape=(HParams.MAX_SENTENCE_DIM, HParams.getFeatureExtractorDim(), 1))
    model(input)
    print(f"input:{(HParams.MAX_SENTENCE_DIM, HParams.getFeatureExtractorDim(), 1)}")
    model.summary()
