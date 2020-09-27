from typing import List

import numpy as np
import tensorflow as tf

# from tqdm.auto import tqdm # Uncomment for Colab-Notebook
from dataprocess.cleaners import cleanQuery
from hparams import HParams
from models.DabaCnnAutoencoder import DabaCnnAutoencoder
from models.SimpleCnnAutoencoder import SimpleCnnAutoencoder
from models.SimpleFCNAutoencoder import SimpleFCNAutoencoder
from models.YabaDabaDiscriminator import DabaDiscriminator
from models.utils import dec2bin


def toBinaryRepresentation(arr):
    res = []
    for num in arr:
        res += dec2bin(num)
    return np.asarray(res)


def toBinaryThreshold(arr, threshold=0.5):
    mask = arr > threshold
    arr[mask] = 1
    arr[np.logical_not(mask)] = 0
    return arr.flatten()


class NNHashEncoder(object):
    @staticmethod
    def get_train_path(model, train_range):
        return f"./checkpoints/train_{str(model)}_train_size_{str(train_range)}"

    def __init__(self, model, discriminator, featureExtractor, optimizer=tf.optimizers.Adam(), restore_last=False,
                 chkp_path=None):
        self.featureExtractor = featureExtractor
        self.optimizer = optimizer
        self.model = model
        self.discriminator = discriminator
        self.chkp_path = chkp_path or NNHashEncoder.get_train_path(self.model, HParams.TRAIN_DATASET_RANGE)
        self.ckpt_manager = self.load(restore_last, chkp_path)

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

    def load(self, restore_last=True, chkp_path=None):
        # checkpoint_path = chkp_path or f"./checkpoints/train_{str(self.model)}"
        checkpoint_path = chkp_path
        ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if restore_last:
            print(f'-- loading weights from {checkpoint_path}')
            save_path = ckpt_manager.latest_checkpoint or tf.train.latest_checkpoint(checkpoint_path)
            if save_path is not None:
                ckpt.restore(save_path).expect_partial()
                # expect_partial due to that we don't need the decode part later on.
                print('Latest checkpoint restored!!')

        return ckpt_manager

    def save(self):
        self.ckpt_manager.save()


def getNNHashEncoder(restore_last=True, skip_discriminator=False):
    return getNNHashEncoder_New(restore_last=restore_last,
                                model_type=HParams.MODEL_TYPE,
                                train_range=HParams.TRAIN_DATASET_RANGE)


LAST_NNHashEncoder = None
LAST_CHKP_PATH = None


def reuse(chkp_path: str):
    if chkp_path is not None and LAST_CHKP_PATH == chkp_path:
        return LAST_NNHashEncoder
    return None


def getNNHashEncoder_New(restore_last=True, model_type=HParams.MODEL_TYPE, train_range=HParams.TRAIN_DATASET_RANGE):
    featureExtractor = HParams.getFeatureExtractor()
    models = {
        'DABA': lambda: DabaCnnAutoencoder(featureExtractor.get_feature_dim(), HParams.OUTPUT_DIM),
        'CNN': lambda: SimpleCnnAutoencoder(featureExtractor.get_feature_dim(), HParams.OUTPUT_DIM),
        'FCN': lambda: SimpleFCNAutoencoder(featureExtractor.get_feature_dim(), HParams.OUTPUT_DIM),
    }
    model = models[model_type]()
    if HParams.MODEL_MODE == 'GAN':
        discriminator = DabaDiscriminator()
    else:
        discriminator = None
    train_path = NNHashEncoder.get_train_path(model, train_range)
    reuse_encoder = reuse(train_path)
    if reuse_encoder:
        print(f"Reuse:{train_path}")
        return reuse_encoder
    else:
        encoder = NNHashEncoder(model, discriminator, featureExtractor, restore_last=restore_last, chkp_path=train_path)
        global LAST_NNHashEncoder, LAST_CHKP_PATH
        LAST_NNHashEncoder = encoder
        LAST_CHKP_PATH = train_path
        return encoder
