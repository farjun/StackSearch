import os
import pickle
from datasketch import MinHash, MinHashLSH
from typing import List

from hparams import HParams
import json

from index.utils import createDirIfNotExists


class MinHashIndex(object):

    def __init__(self, indexPath, overwrite=False, hash_func=None, threshold=0.5):
        self.indexPath = indexPath
        createDirIfNotExists(indexPath)
        self.indexPickleFilePath = os.path.join(indexPath, "main_index_new")
        self._hash_func = hash_func
        self._threshold = threshold
        self.configFilePath = os.path.join(indexPath, "config")
        self.config = dict(indexSize=0)
        self.lsh = None
        if os.path.exists(self.indexPickleFilePath) and not overwrite:
            self.loadIndex()
        else:
            self.initNewIndex()

    def initNewIndex(self):
        self.lsh = MinHashLSH(threshold=self._threshold, num_perm=128)
        os.makedirs(os.path.dirname(self.indexPickleFilePath), exist_ok=True)
        with open(self.indexPickleFilePath, 'wb') as f:
            f.write(b"")  # create file
        os.makedirs(os.path.dirname(self.configFilePath), exist_ok=True)
        with open(self.configFilePath, "w") as f:
            f.write("")  # create file

    def loadIndex(self):
        with open(self.indexPickleFilePath, "rb") as f:
            self.lsh = pickle.load(f)
        with open(self.configFilePath, "r") as f:
            try:
                self.config = json.loads(f.readline())
            except EOFError:
                print("Warning, Config was not able to load from an empty config file")

    def sentence_minhash(self, text: List[str]):
        m = MinHash(num_perm=128) if self._hash_func is None else MinHash(num_perm=128, hashfunc=self._hash_func)
        for word in text:
            m.update(word.encode('utf8'))
        return m

    def insert(self, post_id, text: List[str]):
        m = self.sentence_minhash(text)
        self.lsh.insert(post_id, m)
        self.config['indexSize'] += 1

    def search(self, text: List[str], top_k=10):
        m = self.sentence_minhash(text)
        result = self.lsh.query(m)
        return result[:top_k]

    def size(self):
        return self.config['indexSize']

    def save(self):
        data = pickle.dumps(self.lsh)
        with open(self.indexPickleFilePath, 'wb') as f:
            f.write(data)
        with open(self.configFilePath, "w") as f:
            f.write(json.dumps(self.config))


