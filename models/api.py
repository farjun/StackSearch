from typing import List

import numpy as np
import tensorflow as tf

# from tqdm.auto import tqdm # Uncomment for Colab-Notebook
from dataprocess.cleaners import cleanQuery
from hparams import HParams
from models.SimpleCnnAutoencoder import SimpleCnnAutoencoder
from models.SimpleFCNAutoencoder import SimpleFCNAutoencoder
from models.YabaDabaDiscriminator import DabaDiscriminator
from models.utils import dec2bin


def toBinaryRepresentation(arr):
    res = []
    for num in arr:
        res += dec2bin(num)
    return np.asarray(res)


def toBinaryThreshold(arr, threshold = 0.5):
    mask = arr > threshold
    arr[mask] = 1
    arr[np.logical_not(mask)] = 0
    return arr.flatten()

class NNHashEncoder(object):
    def __init__(self, model, discriminator, featureExtractor, optimizer=tf.optimizers.Adam(), restore_last=False):
        self.featureExtractor = featureExtractor
        self.optimizer = optimizer
        self.model = model
        self.discriminator = discriminator
        self.ckpt_manager = self.load(restore_last)

    def encode_batch(self, words: List[str]):
        encode = self.encode_batch_no_mask(words)
        # return toBinaryRepresentation(encode)
        return encode

    def encode_batch_no_mask(self, words):
        feature = self.featureExtractor.get_feature_batch(words)
        feature = np.array(feature)
        feature = feature[tf.newaxis, ...]
        encode = self.model.encode(feature)
        encode = encode.numpy()
        return encode.flatten()

    def clean_and_encode_query(self, words: List[str]):
        return self.encode_batch(cleanQuery(words))

    def load(self, restore_last=True):
        checkpoint_path = f"./checkpoints/train_{str(self.model)}"
        ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if restore_last:
            save_path = ckpt_manager.latest_checkpoint or tf.train.latest_checkpoint(checkpoint_path)
            if save_path is not None:
                ckpt.restore(save_path).expect_partial()
                # expect_partial due to that we don't need the decode part later on.
                print('Latest checkpoint restored!!')

        return ckpt_manager

    def save(self):
        self.ckpt_manager.save()


def getNNHashEncoder(restore_last=True,skip_discriminator=False):
    featureExtractor = HParams.getFeatureExtractor()
    # model = DabaCnnAutoencoder(featureExtractor.get_feature_dim(), HParams.OUTPUT_DIM)
    model = SimpleCnnAutoencoder(featureExtractor.get_feature_dim(), HParams.OUTPUT_DIM)
    # model = SimpleFCNAutoencoder(featureExtractor.get_feature_dim(), HParams.OUTPUT_DIM)
    if not skip_discriminator:
        discriminator = DabaDiscriminator()
    else:
        discriminator = None
    return NNHashEncoder(model, discriminator, featureExtractor, restore_last=restore_last)

