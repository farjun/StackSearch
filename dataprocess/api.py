from random import random

from Features.FeatureExtractors import FeatureExtractor, HistogramFeatureExtractor, NNWordEmbeddingFeatureExtractor
import tensorflow as tf

from dataprocess.parser import XmlParser
from hparams import HParams


def example_get_data_generator(featureExtractor: FeatureExtractor):
    size = 1000
    stop_prob = 0.1
    words = []
    random_chr = lambda: chr(random.randint(ord('a'), ord('z')))
    for i in range(size):
        word = random_chr()
        while random.random() > stop_prob:
            word += random_chr()
        words.append(word)
    as_features = featureExtractor.get_feature_batch(words)

    def to_generator():
        for features in as_features:
            yield features

    return to_generator


def get_data_set_example(featureExtractor: FeatureExtractor):
    ds = tf.data.Dataset.from_generator(example_get_data_generator(featureExtractor), (tf.int32))
    ds = ds \
        .cache() \
        .batch(HParams.BATCH_SIZE) \
        .shuffle(10000) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_partial_data_set(featureExtractor: FeatureExtractor):
    xmlParser = XmlParser(HParams.filePath)
    ds = tf.data.Dataset.from_generator(xmlParser.getWordsGenerator(featureExtractor=featureExtractor), (tf.int32))
    ds = ds \
        .cache() \
        .batch(HParams.BATCH_SIZE) \
        .shuffle(10000) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def get_partial_data_set_titles(featureExtractor: FeatureExtractor):
    xmlParser = XmlParser(HParams.filePath)
    ds = tf.data.Dataset.from_generator(xmlParser.getTitleGenerator(featureExtractor=featureExtractor), (tf.int32))
    ds = ds \
        .cache() \
        .batch(HParams.BATCH_SIZE) \
        .shuffle(10000) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def resolve_data_set(dataset_type: str, featureExtractor = NNWordEmbeddingFeatureExtractor()):
    default = "example"
    types = {
        default: get_data_set_example,
        "partial": get_partial_data_set,
        "partial_titles": get_partial_data_set_titles
    }
    if dataset_type is None:
        dataset_type = default
    if dataset_type not in types:
        print(f"dataset_type:{dataset_type} not in supported keys: {types.keys()}")
        raise NotImplementedError
    return types[dataset_type](featureExtractor)