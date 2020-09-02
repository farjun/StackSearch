from typing import List
import numpy as np
from dataprocess.models import Post
from nltk.corpus import brown
from gensim.models import Word2Vec, Doc2Vec
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

    def get_feature(self, word: str):
        histogram = [0] * self.get_feature_dim()
        for c in word:
            if ord(c) - ord('a') in range(len(histogram)):
                histogram[ord(c) - ord('a')] += 1
        return np.array(histogram, dtype=np.float32)

    def get_feature_batch(self, words: List[str], maxSentenceDim = HParams.MAX_SENTENCE_DIM) -> np.ndarray:
        result = np.zeros((maxSentenceDim, self.get_feature_dim()), dtype=np.float32)
        for i,word in enumerate(words):
            wordVec = self.get_feature(word)
            if np.max(wordVec) != 0:
                wordVec = wordVec / np.max(wordVec)
            result[i,:] = wordVec
        result = result[..., tf.newaxis]
        return result


class W2VFeatureExtractor(FeatureExtractor):

    def __init__(self, dim=None):
        # see https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py
        self.dim = dim or (HParams.MAX_SENTENCE_DIM * self.get_feature_dim())
        self.model = None
        self._path = HParams.word2vecFilePath
        if os.path.exists(self._path):
            self.model = Word2Vec.load(self._path)
        else:
            raise Exception('call train_embedding_word2vec from models/train.py')

    def get_feature_dim(self):
        return ord('z') - ord('a') + 7

    def get_feature(self, word: str):
        missing_word_case = np.array([0.0001] * self.dim)
        if word in self.model.wv:
            return self.model.wv[word]
        return missing_word_case

    def get_feature_batch(self, words: List[str], maxSentenceDim=HParams.MAX_SENTENCE_DIM) -> np.ndarray:
        sum_vec = np.zeros(self.dim, dtype=np.float32)
        for word in words:
            word_vec = self.get_feature(word)
            sum_vec += word_vec
        result = sum_vec / len(words)
        result.resize((maxSentenceDim, self.get_feature_dim(), 1))
        return result


class D2VFeatureExtractor(FeatureExtractor):

    def __init__(self, dim=None):
        # see https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py
        self.dim = dim or (HParams.MAX_SENTENCE_DIM * self.get_feature_dim())
        self.model = None
        self._path = HParams.doc2vecFilePath
        if os.path.exists(self._path):
            self.model = Doc2Vec.load(self._path)
        else:
            raise Exception('call train_embedding_doc2vec from models/train.py')

    def get_feature_dim(self):
        return ord('z') - ord('a') + 7

    def get_feature(self, word: str):
        # this method shouldn't be in use
        return self.get_feature_batch([word])

    def get_feature_batch(self, words: List[str], maxSentenceDim=HParams.MAX_SENTENCE_DIM) -> np.ndarray:
        result = self.model.infer_vector(words)
        result.resize((maxSentenceDim, self.get_feature_dim(), 1))
        return result
