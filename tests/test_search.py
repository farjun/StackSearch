import unittest
import logging

from hparams import HParams
from index.index import MinHashIndex
from tempfile import TemporaryDirectory
import numpy as np
from models.api import getNNHashEncoder
from dataprocess.cleaners import cleanQuery
from main import saveYabaDabaIndex

def padVec(vec):
    return np.pad(np.array(vec),(0,HParams.OUTPUT_DIM - len(vec)))

class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.index = MinHashIndex(self.tmp_dir.name, overwrite=True)
        self.encoder = getNNHashEncoder()
        self.encoder.load(restore_last=True)

    def _checkExpectedInQuery(self, toSearch, expectedId, topK = 10):
        toSearch = cleanQuery(toSearch)
        res = self.index.search(self.encoder.encode_batch(toSearch), top_k=topK)
        return expectedId in res

    def test_search_and_insert(self):
        self.index.insert(1, self.encoder.clean_and_encode_query("i have a pen i have an apple"))
        self.index.insert(2, self.encoder.clean_and_encode_query("numpy python"))
        self.index.insert(3, self.encoder.clean_and_encode_query("please work"))
        self.index.insert(4, self.encoder.clean_and_encode_query("maple story good game"))

        self.index.index()
        self.assertTrue(self._checkExpectedInQuery("i have a pen i have an apple", 1))
        self.assertTrue(self._checkExpectedInQuery("numpy python matrix", 2))
        self.assertTrue(self._checkExpectedInQuery("maple story good game", 4))


    def tearDown(self) -> None:
        pass

if __name__ == '__main__':
    unittest.main()
