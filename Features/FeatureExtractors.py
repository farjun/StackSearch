from typing import List
import numpy as np
from dataprocess.models import Post
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
        return ord('z') - ord('a') + 1

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
            result[i,:] = wordVec / np.max(wordVec)
        result = result[..., tf.newaxis]
        return result