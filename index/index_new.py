import heapq
import os
import pickle
from datasketch import MinHash, MinHashLSH
from typing import List, Union

from dataprocess.parser import XmlParser
from hparams import HParams
import json

from index.utils import createDirIfNotExists


class MinHashIndex(object):

    def __init__(self, indexPath, overwrite=False, hash_func=None, threshold=0.5, num_perm=128):
        self.indexPath = indexPath
        self.num_perm = num_perm
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
        self.lsh = MinHashLSH(threshold=self._threshold, num_perm=self.num_perm)
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

    def sentence_minhash(self, text: Union[List[str], str]):
        m = MinHash(num_perm=self.num_perm) if self._hash_func is None else MinHash(num_perm=self.num_perm,
                                                                                    hashfunc=self._hash_func)
        text = [text] if isinstance(text, str) else text
        for word in text:
            m.update(word.encode('utf8'))
        return m

    def insert(self, post_id, text: Union[List[str], str]):
        if isinstance(text, str):
            text = [text]
        m = self.sentence_minhash(text)
        self.lsh.insert(post_id, m)
        self.config['indexSize'] += 1

    def search(self, text: Union[List[str], str], result_limit=10):
        m = self.sentence_minhash(text)
        result = self.lsh.query(m)
        if len(result) > result_limit:
            post_title_tups = [(postId, XmlParser.getPostTitle(postId)) for postId in result]
            result = heapq.nlargest(result_limit, post_title_tups, key=lambda post_title_tup: self.compouteJaccardSim(m, post_title_tup[1]))
            result = [_[0] for _ in result]

        return result[:result_limit]

    def compouteJaccardSim(self, m, title : str):
        return self.sentence_minhash(title).jaccard(m)

    def size(self):
        return self.config['indexSize']

    def save(self):
        data = pickle.dumps(self.lsh)
        with open(self.indexPickleFilePath, 'wb') as f:
            f.write(data)
        with open(self.configFilePath, "w") as f:
            f.write(json.dumps(self.config))


