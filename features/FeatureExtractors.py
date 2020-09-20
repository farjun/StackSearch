from typing import List
import numpy as np
from dataprocess.models import Post
from nltk.corpus import brown
from gensim.models import Word2Vec, Doc2Vec
import os
from hparams import HParams
from os import path
import hparams
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


class WordEmbeddingToMatrixFeatureExtractor(HistogramFeatureExtractor):
    def __init__(self, numOfWordsToDrop=0):
        super().__init__()
        self.numOfWordsToDrop = numOfWordsToDrop

    def get_feature(self, word: str):
        histogram = [0] * self.get_feature_dim()
        for c in word:
            if ord(c) - ord('a') in range(len(histogram)):
                histogram[ord(c) - ord('a')] += 1
        return np.array(histogram, dtype=np.float32)

    def get_feature_batch(self, words: List[str]) -> np.ndarray:
        result = np.zeros((hparams.HParams.MAX_SENTENCE_DIM, self.get_feature_dim()), dtype=np.float32)
        for i, word in enumerate(words):
            wordVec = self.get_feature(word)
            if np.max(wordVec) != 0:
                wordVec = wordVec / np.max(wordVec)
                wordVec = np.around(wordVec)
            result[i, :] = wordVec

        if self.numOfWordsToDrop > 0:
            result[np.random.choice(result.shape[0], size=self.numOfWordsToDrop)] = 0

        result = result[..., tf.newaxis]
        return result

    def get_noised_feature_batch(self, words: List[str], numOfWordsToDrop=2):
        features = self.get_feature_batch(words)
        features[np.random.choice(features.shape[0], size=numOfWordsToDrop)] = 0
        return features


class W2VFeatureExtractor(FeatureExtractor):

    def __init__(self, dim=None, numOfWordsToDrop=0):
        # see https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py
        self.dim = dim or (hparams.HParams.MAX_SENTENCE_DIM * self.get_feature_dim())
        self.model = None
        self._path = os.path.join(hparams.HParams.embeddingFilePath, f"word2v_embedding_{numOfWordsToDrop}")
        self.config = dict(minval=0, maxval=0)
        if os.path.exists(self._path):
            self.model = Word2Vec.load(self._path)
        else:
            pass

    def get_feature_dim(self):
        return 32

    def get_feature(self, word: str):
        if word in self.model.wv:
            return self.model.wv[word]
        missing_word_case = np.array([0.0001] * self.dim)
        return missing_word_case

    def _updateminmax(self, result):
        if self.config['minval'] > result.min():
            self.config['minval'] = result.min()
            print(self.config)

        if self.config['maxval'] < result.max():
            self.config['maxval'] = result.max()
            print(self.config)

    def get_feature_batch(self, words: List[str], maxSentenceDim=hparams.HParams.MAX_SENTENCE_DIM) -> np.ndarray:
        if len(words) == 0:
            return np.zeros(self.dim, dtype=np.float32)
        sum_vec = np.zeros(self.dim, dtype=np.float32)
        for word in words:
            word_vec = self.get_feature(word)
            sum_vec += word_vec

        result = sum_vec / len(words)
        # self._updateminmax(result)
        result.resize((maxSentenceDim, self.get_feature_dim(), 1))
        return np.interp(result, (result.min(), result.max()), (-0.00048781678, 0.00048719026))


class FeatureExtractor_Temp(FeatureExtractor):
    # TODO this is a temp class which immitate D2V with W2V

    def __init__(self, dim=200, numOfWordsToDrop=0):
        self.dim = dim
        self.numOfWordsToDrop = numOfWordsToDrop
        self.model = None
        self._path = os.path.join(hparams.HParams.embeddingFilePath, f"word2v_embedding")
        if os.path.exists(self._path):
            self.model = Word2Vec.load(self._path)
        else:
            pass
            # raise Exception('call train_embedding_doc2vec from models/train.py')

    def get_feature_dim(self):
        return self.dim

    def get_feature(self, word: str):
        if word in self.model.wv:
            return self.model.wv[word]
        missing_word_case = np.full(self.dim, 0.0001)
        return missing_word_case

    def get_feature_batch(self, words: List[str], maxSentenceDim=hparams.HParams.MAX_SENTENCE_DIM) -> np.ndarray:
        if len(words) == 0:
            raise Exception("0 length sentence.")
        out = np.zeros(shape=(maxSentenceDim, self.get_feature_dim()))
        for i in range(min(len(words), maxSentenceDim)):
            out[i] = self.get_feature(words[i])
        if self.numOfWordsToDrop > 0:
            print(self.numOfWordsToDrop)
            indexes = np.random.choice(maxSentenceDim, size=self.numOfWordsToDrop, replace=False)
            out[indexes] = 0
        result = out[..., np.newaxis]
        # result = np.interp(result, (result.min(), result.max()), (-0.00048781678, 0.00048719026))

        return result


class D2VFeatureExtractor(FeatureExtractor):

    def __init__(self, dim=None, numOfWordsToDrop=0):
        # see https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py
        self.dim = dim or (hparams.HParams.MAX_SENTENCE_DIM * self.get_feature_dim())
        self.model = None

        self._path = os.path.join(hparams.HParams.embeddingFilePath, f"word2v_embedding_{numOfWordsToDrop}")
        if os.path.exists(self._path):
            self.model = Doc2Vec.load(self._path)
        else:
            raise Exception('call train_embedding_doc2vec from models/train.py')

    def get_feature_dim(self):
        return ord('z') - ord('a') + 7

    def get_feature(self, word: str):
        # this method shouldn't be in use
        return self.get_feature_batch([word])

    def get_feature_batch(self, words: List[str], maxSentenceDim=hparams.HParams.MAX_SENTENCE_DIM) -> np.ndarray:
        if len(words) == 0:
            raise Exception("0 length sentence.")
        result = self.model.infer_vector(words)
        result.resize((maxSentenceDim, self.get_feature_dim(), 1))
        return result


if __name__ == '__main__':
    ret = frozenset(np.random.choice(100, size=99, replace=False))
    for i in range(100):
        if i not in ret:
            print(i)
