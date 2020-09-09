import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2DTranspose, Dense, Reshape, BatchNormalization

from hparams import HParams


class SimpleCnnAutoencoder(tf.keras.Model):
    def __init__(self, featureDim, latent_space_dim, useNormalization = True, *args, **kwargs):
        self.useNormalization = useNormalization
        assert featureDim % 4 == 0
        assert HParams.MAX_SENTENCE_DIM % 4 == 0

        self.inputshape = (HParams.MAX_SENTENCE_DIM, featureDim)
        self.latent_space_dim = latent_space_dim
        super().__init__(*args, **kwargs)

        if self.useNormalization:
            self.batchNormalization0 = BatchNormalization()
            self.batchNormalization1 = BatchNormalization()
            self.batchNormalization2 = BatchNormalization()
            self.batchNormalization3 = BatchNormalization()
            self.batchNormalization4 = BatchNormalization()

        #encode
        self.down_c1 = tf.keras.layers.Conv2D(300, kernel_size=(5, latent_space_dim), strides=2, padding='same', activation='relu')
        self.down_c2 = tf.keras.layers.Conv2D(600, kernel_size=(5, 1), strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.down_d1 = tf.keras.layers.Dense(latent_space_dim, activation="sigmoid")

        #decode
        self.up_d1 = Dense(512, activation='relu')
        self.up_d2 = Dense(self.inputshape[0]/4 * self.inputshape[1]/4 * 64, activation='relu')
        self.reshaper = Reshape((int(self.inputshape[0]/4) , int(self.inputshape[1]/4), 64))
        self.up_ct1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')
        self.up_ct2 = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')


    def call(self, inputs, training=None, mask=None):
        return self.decode(self.encode(inputs, training), training)

    def encode(self, inputs, training=False):
        x = self.batchNormalization0(inputs)
        x = self.down_c1(x, training = training)
        x = self.batchNormalization1(x, training = training) if self.useNormalization else x
        x = self.down_c2(x, training = training)
        x = self.batchNormalization2(x, training = training) if self.useNormalization else x
        x = self.flatten(x)
        x = self.down_d1(x, training = training)
        return x

    def decode(self, inputs, training=False):
        x = inputs
        x = self.up_d1(x, training=training)
        x = self.batchNormalization3(x, training = training) if self.useNormalization else x
        x = self.up_d2(x, training=training)
        x = self.batchNormalization4(x, training = training) if self.useNormalization else x
        x = self.reshaper(x)
        x = self.up_ct1(x, training=training)
        x = self.up_ct2(x, training=training)
        return x

    def __str__(self):
        return "SimpleCnnAutoencoder_" + str(self.latent_space_dim)


if __name__ == '__main__':

    model = SimpleCnnAutoencoder(HParams.getFeatureExtractorDim(), HParams.OUTPUT_DIM)
    input = tf.keras.layers.Input(shape=(HParams.MAX_SENTENCE_DIM, HParams.getFeatureExtractorDim(), 1))
    model(input)
    print(f"input:{(HParams.MAX_SENTENCE_DIM, HParams.getFeatureExtractorDim(), 1)}")
    model.summary()
