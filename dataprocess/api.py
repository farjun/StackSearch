import numpy as np
from random import random

from features.FeatureExtractors import FeatureExtractor
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


def get_data_set_example(featureExtractor: FeatureExtractor, amount_to_drop):
    ds = tf.data.Dataset.from_generator(example_get_data_generator(featureExtractor), (tf.int32))
    ds = ds \
        .cache() \
        .batch(HParams.BATCH_SIZE) \
        .shuffle(10000) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_partial_data_set(featureExtractor: FeatureExtractor, *args, **kwargs):
    xmlParser = XmlParser(HParams.filePath)
    ds = tf.data.Dataset.from_generator(xmlParser.getWordsGenerator(featureExtractor=featureExtractor), (tf.int32))
    ds = ds \
        .cache() \
        .batch(HParams.BATCH_SIZE) \
        .shuffle(10000) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_data_set_titles(featureExtractor: FeatureExtractor, amount_to_drop):
    xmlParser = XmlParser(HParams.filePath)
    ds = tf.data.Dataset.from_generator(xmlParser.getTitleGenerator(featureExtractor=featureExtractor), (tf.float32),
                                        output_shapes=(HParams.MAX_SENTENCE_DIM, featureExtractor.get_feature_dim(), 1))
    ds = ds \
        .shuffle(100) \
        .batch(HParams.BATCH_SIZE) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def drop_some_words_numpy_batch(value, amount_to_drop):
    # Assuming batch
    batch_size = value.shape[0]  # this is not a constant (see drop_reminder)
    sentence_dim = value.shape[1]
    for i in range(batch_size):
        mask = np.random.choice(sentence_dim, size=amount_to_drop, replace=False)
        value[i, mask] = 0
    return value


def swap_some_words_numpy_batch(x, x_hat, amount_to_swap):
    # Assuming batch
    batch_size = x.shape[0]  # this is not a constant (see drop_reminder)
    sentence_dim = x.shape[1]
    for i in range(batch_size):
        mask = np.random.choice(sentence_dim, size=amount_to_swap, replace=False)
        mask_permutation = np.random.permutation(mask)
        x[i, mask] = x[i, mask_permutation]
        x_hat[i, mask] = x_hat[i, mask_permutation]
    return np.array([x, x_hat])


def drop_some_words(amount_to_drop):
    def inner(x: tf.Tensor, x_hat: tf.Tensor):
        ret = tf.numpy_function(drop_some_words_numpy_batch, [x, amount_to_drop], x.dtype)
        ret.set_shape(x.shape)
        return ret, x_hat

    return inner


def swap_some_words(amount_to_swap):
    def inner(x: tf.Tensor, x_hat: tf.Tensor):
        ret = tf.numpy_function(swap_some_words_numpy_batch, [x, x_hat, amount_to_swap], x.dtype)
        ret, ret_hat = ret[0], ret[1]
        ret.set_shape(x.shape)
        ret_hat.set_shape(x_hat.shape)
        return ret, ret_hat

    return inner


def temp_f(featureExtractor: FeatureExtractor, amount_to_drop, amount_to_swap):
    xmlParser = XmlParser(HParams.filePath)
    output_shapes = (HParams.MAX_SENTENCE_DIM, featureExtractor.get_feature_dim(), 1)
    ds = tf.data.Dataset.from_generator(xmlParser.getTitleGenerator(featureExtractor=featureExtractor), (tf.float32),
                                        output_shapes=output_shapes)
    ds = ds.cache()
    ds = ds.shuffle(100)
    ds = ds.batch(HParams.BATCH_SIZE)  # Efficient
    ds = ds.map(lambda x: (x, x))  # duplicate to create x,x_hat
    if amount_to_drop > 0:
        ds = ds.map(drop_some_words(amount_to_drop), num_parallel_calls=10)
    if amount_to_swap > 0:
        ds = ds.map(swap_some_words(amount_to_swap), num_parallel_calls=10)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def resolve_data_set(dataset_type: str, amount_to_drop=0, amount_to_swap=0):
    default = "example"
    types = {
        default: get_data_set_example,
        "partial": get_partial_data_set,
        "titles": temp_f
    }
    if dataset_type is None:
        dataset_type = default
    if dataset_type not in types:
        print(f"dataset_type:{dataset_type} not in supported keys: {types.keys()}")
        raise NotImplementedError
    return types[dataset_type](HParams.getFeatureExtractor(), amount_to_drop, amount_to_swap)


if __name__ == '__main__':
    ds = temp_f(HParams.getFeatureExtractor(), 5, 5)
    for item, item2 in ds:
        print(item)
