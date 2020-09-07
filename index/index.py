import os
import pickle
from datasketch import WeightedMinHashGenerator, MinHashLSHForest, WeightedMinHash, LeanMinHash
from hparams import HParams
import json

class Index(object):
    def __init__(self, indexPath):
        self.indexPath = indexPath


class MinHashIndex(Index):

    def __init__(self, indexPath, overwrite=False):
        super().__init__(indexPath)
        self.indexPickleFilePath = os.path.join(indexPath, "main_index")
        self.minHashGeneratorPickleFilePath = os.path.join(indexPath, "min_hash_gen")
        self.configFilePath = os.path.join(indexPath, "config")
        self.config = dict( indexSize = 0 )
        if os.path.exists(self.indexPickleFilePath) and not overwrite:
            self.loadIndex()
        else:
            self.initNewIndex()

    def initNewIndex(self):
        self.minHashGenerator = WeightedMinHashGenerator(HParams.OUTPUT_DIM)
        self.hasher = MinHashLSHForest(num_perm=HParams.OUTPUT_DIM,
                                       l=1)  # performs the document hashing and results using Min Hash
        os.makedirs(os.path.dirname(self.indexPickleFilePath),exist_ok=True)
        with open(self.indexPickleFilePath, 'wb') as f:
            f.write(b"")  # create file
        os.makedirs(os.path.dirname(self.minHashGeneratorPickleFilePath),exist_ok=True)
        with open(self.minHashGeneratorPickleFilePath, "wb") as f:
            f.write(b"")  # create file
        os.makedirs(os.path.dirname(self.configFilePath),exist_ok=True)
        with open(self.configFilePath, "w") as f:
            f.write("")  # create file

    def loadIndex(self):
        with open(self.indexPickleFilePath, "rb") as f:
            self.hasher = pickle.load(f)
        with open(self.minHashGeneratorPickleFilePath, "rb") as f:
            self.minHashGenerator = pickle.load(f)
        with open(self.configFilePath, "r") as f:
            try:
                self.config = json.loads(f.readline())
            except EOFError:
                print("Warning, Config was not able to load from an empty config file")

    def insert(self, postId, vec):
        self.hasher.add(postId, WeightedMinHash(1 , vec))
        self.config['indexSize'] += 1

    def search(self, vec, top_k=2):
        return self.hasher.query(WeightedMinHash(1 , vec), top_k)

    def index(self):
        self.hasher.index()

    def size(self):
        return self.config['indexSize']

    def save(self):
        data = pickle.dumps(self.hasher)
        with open(self.indexPickleFilePath, 'wb') as f:
            f.write(data)
        data = pickle.dumps(self.minHashGenerator)
        with open(self.minHashGeneratorPickleFilePath, 'wb') as f:
            f.write(data)
        with open(self.configFilePath, "w") as f:
            f.write(json.dumps(self.config))
