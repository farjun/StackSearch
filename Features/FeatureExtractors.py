from typing import List
import numpy as np
from dataprocess.models import Post
from gensim.models import Word2Vec
from nltk.corpus import brown
import gensim
import os

from hparams import HParams
import tensorflow as tf


class FeatureExtractor:
    def get_feature_dim(self):
        raise NotImplementedError

    def get_feature(self, word) -> List:
        raise NotImplementedError

    def get_feature_batch(self, words: List[str]) -> List[List]:
        result = []
        for word in words:
            result.append(self.get_feature(word))
        return result


class HistogramFeatureExtractor(FeatureExtractor):

    def __init__(self):
        pass

    def get_feature_dim(self):
        return ord('z') - ord('a') + 7

    def get_feature(self, word: str):
        histogram = [0] * self.get_feature_dim()
        for c in word:
            if ord(c) - ord('a') in range(len(histogram)):
                histogram[ord(c) - ord('a')] += float(1.0)
        return histogram


class NNWordEmbeddingFeatureExtractor(HistogramFeatureExtractor):
    def __init__(self, numOfWordsToDrop = 0):
        super()
        self.numOfWordsToDrop = numOfWordsToDrop

    def get_feature(self, word: str):
        histogram = [0] * self.get_feature_dim()
        for c in word:
            if ord(c) - ord('a') in range(len(histogram)):
                histogram[ord(c) - ord('a')] += 1
        return np.array(histogram, dtype=np.float32)

    def get_feature_batch(self, words: List[str]) -> np.ndarray:
        result = np.zeros((HParams.MAX_SENTENCE_DIM, self.get_feature_dim()), dtype=np.float32)
        for i,word in enumerate(words):
            wordVec = self.get_feature(word)
            if np.max(wordVec) != 0:
                wordVec = wordVec / np.max(wordVec)
            result[i,:] = wordVec

        if self.numOfWordsToDrop > 0:
            result[np.random.choice(result.shape[0], size=self.numOfWordsToDrop)] = 0

        result = result[..., tf.newaxis]
        return result

    def get_noised_feature_batch(self, words:List[str], numOfWordsToDrop = 2):
        features = self.get_feature_batch(words)
        features[np.random.choice(features.shape[0], size=numOfWordsToDrop)] = 0
        return features

class AdvFeatureExtractor(FeatureExtractor):

    def __init__(self, dim=26):
        self.dim = dim
        self.model = None
        self._path = './checkpoints/word2vec/brown_1'
        if os.path.exists(self._path):
            self.model = gensim.models.Word2Vec.load(self._path)
        else:
            self.train()

    def get_feature_dim(self):
        return self.dim

    def train(self):
        #TODO train on our data as well
        self.model = gensim.models.Word2Vec(brown.sents(), size=self.dim)
        self.model.save(self._path)

    def get_feature(self, word: str):
        missing_word_case = np.array([0.0001] * self.dim)
        if word in self.model.wv:
            return self.model.wv[word]
        return missing_word_case


