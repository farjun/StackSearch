from typing import List

import numpy as np
import tensorflow as tf

# from tqdm.auto import tqdm # Uncomment for Colab-Notebook
from Features.FeatureExtractors import HistogramFeatureExtractor, NNWordEmbeddingFeatureExtractor
from hparams import HParams
from models.DabaCnnAutoencoder import DabaCnnAutoencoder


class NNHashEncoder(object):
    def __init__(self, model, featureExtractor, optimizer = tf.optimizers.Adam(), restore_last=False):
        self.featureExtractor = featureExtractor
        self.optimizer = optimizer
        self.model = model

        self.ckpt_manager = self.load(restore_last)

    def encode(self, word: str):
        feature = np.expand_dims(np.array(self.featureExtractor.get_feature(word)), 0)
        encode = self.model.encode(feature)[0]
        encode = encode.numpy()
        mask = encode > 0
        encode[mask] = 1
        encode[np.logical_not(mask)] = 0
        return encode


    def encode_batch(self, words: List[str]):
        feature = np.array(self.featureExtractor.get_feature_batch(words))
        encode = self.model.encode(feature)
        encode = encode.numpy()
        mask = encode > 0
        encode[mask] = 1
        encode[np.logical_not(mask)] = 0
        return encode


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

def getNNHashEncoder(restore_last=True):
    featureExtractor = NNWordEmbeddingFeatureExtractor()
    model = DabaCnnAutoencoder(featureExtractor.get_feature_dim(), HParams.OUTPUT_DIM)
    return NNHashEncoder(model, featureExtractor, restore_last = restore_last)