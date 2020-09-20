import unittest
import logging

from hparams import HParams
from index.index import MinHashIndex
from tempfile import TemporaryDirectory
import numpy as np
from models.api import getNNHashEncoder
from dataprocess.cleaners import cleanQuery
from main import saveYabaDabaIndex, runSearch
import os

class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        indexPath =  os.path.join(os.path.dirname(HParams.filePath), "index")
        self.index = MinHashIndex(indexPath)
        if self.index.size() != HParams.TRAIN_DATASET_SIZE:
            logging.info("HParams.DATASET_SIZE != index.size() : {} != {}, indexing again".format(HParams.TRAIN_DATASET_SIZE, self.index.size()))
            self.index = saveYabaDabaIndex()

        self.encoder = getNNHashEncoder()
        self.encoder.load(restore_last=True)


    def _checkExpectedInQuery(self, toSearch, expectedId, topK = 10):
        toSearch = cleanQuery(toSearch)
        simHash = self.encoder.encode_batch(toSearch)
        res = self.index.search(simHash, top_k=topK)
        return expectedId in res

    def test_search_and_insert(self):
        self.assertTrue(self._checkExpectedInQuery("What are the preferred versions of Vim and Emacs on Mac OS X", 1496))
        self.assertTrue(self._checkExpectedInQuery("Normalizing a Table with Low Integrity", 6110))
        self.assertTrue(self._checkExpectedInQuery("What is the best way to store connection string in .NET DLLs?", 6113))


    def tearDown(self) -> None:
        pass

if __name__ == '__main__':
    unittest.main()
