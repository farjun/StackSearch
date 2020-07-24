import os

import numpy as np
import pickle
from datasketch import MinHashLSH, WeightedMinHashGenerator
import sklearn
from hparams import HParams

class Index(object):
    def __init__(self, indexPath):
        self.indexPath = indexPath
        self.buffer = list()

    def sort(self):
        pass

class MinHashIndex(Index):
    def __init__(self, indexPath, overwrite = False):
        super().__init__(indexPath)
        self.indexPickleFilePath = os.path.join(indexPath, "main_index")

        if os.path.exists(self.indexPickleFilePath) and not overwrite:
            with open(self.indexPickleFilePath, "rb") as f:
                self.hasher = pickle.load(f)
        else:
            with open(self.indexPickleFilePath, 'wb') as f:
                f.write(b"")  # create file

            self.hasher = MinHashLSH() # performs the document hashing and results using Min Hash

    def getMinHashGenerator(self):
        return WeightedMinHashGenerator(HParams.OUTPUT_DIM)

    def insert(self, postId, vec):
        minHashGenerator = self.getMinHashGenerator()
        self.hasher.insert(postId, minHashGenerator.minhash(vec))

    def search(self, vec):
        minHashGenerator = self.getMinHashGenerator()
        return self.hasher.query(minHashGenerator.minhash(vec))

    def save(self):
        data = pickle.dumps(self.hasher)
        with open(self.indexPickleFilePath, 'wb') as f:
            f.write(data)

