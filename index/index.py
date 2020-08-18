import os
import pickle
from datasketch import WeightedMinHashGenerator, MinHashLSHForest, WeightedMinHash, LeanMinHash
from hparams import HParams


class Index(object):
    def __init__(self, indexPath):
        self.indexPath = indexPath


class MinHashIndex(Index):

    def __init__(self, indexPath, overwrite=False):
        super().__init__(indexPath)
        self.indexPickleFilePath = os.path.join(indexPath, "main_index")
        self.minHashGeneratorPickleFilePath = os.path.join(indexPath, "min_hash_gen")

        if os.path.exists(self.indexPickleFilePath) and not overwrite:
            with open(self.indexPickleFilePath, "rb") as f:
                self.hasher = pickle.load(f)
            with open(self.minHashGeneratorPickleFilePath, "rb") as f:
                self.minHashGenerator = pickle.load(f)
        else:
            self.minHashGenerator = WeightedMinHashGenerator(HParams.OUTPUT_DIM)
            self.hasher = MinHashLSHForest( num_perm = HParams.OUTPUT_DIM, l=1 )  # performs the document hashing and results using Min Hash
            with open(self.indexPickleFilePath, 'wb') as f:
                f.write(b"")  # create file
            with open(self.minHashGeneratorPickleFilePath, "wb") as f:
                f.write(b"")  # create file




    def insert(self, postId, vec):
        self.hasher.add(postId, WeightedMinHash(1 , vec))

    def search(self, vec, top_k=2):
        return self.hasher.query(WeightedMinHash(1 , vec), top_k)

    def index(self):
        self.hasher.index()

    def save(self):
        data = pickle.dumps(self.hasher)
        with open(self.indexPickleFilePath, 'wb') as f:
            f.write(data)
        data = pickle.dumps(self.minHashGenerator)
        with open(self.minHashGeneratorPickleFilePath, 'wb') as f:
            f.write(data)

