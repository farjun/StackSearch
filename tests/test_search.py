import unittest
import logging

from hparams import HParams
from index.index import MinHashIndex
from tempfile import TemporaryDirectory
import numpy as np
from models.api import getNNHashEncoder

def padVec(vec):
    return np.pad(np.array(vec),(0,HParams.OUTPUT_DIM - len(vec)))

class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.index = MinHashIndex(self.tmp_dir.name, overwrite=True)
        self.encoder = getNNHashEncoder()
        self.encoder.load(restore_last=True)

    def test_search_and_insert(self):
        self.index.insert(1, self.encoder.encode_batch("i have a pen i have an apple"))
        self.index.insert(2, self.encoder.encode_batch("numpy python matrix"))
        self.index.insert(3, self.encoder.encode_batch("please work"))
        self.index.insert(4, self.encoder.encode_batch("maple story good game"))

        self.index.index()
        self.assertEqual(self.index.search(self.encoder.encode_batch("apple pen"), top_k=2), [1])
        self.assertEqual(self.index.search(self.encoder.encode_batch("numpy python"), top_k=1), [2])
        self.assertEqual(self.index.search(self.encoder.encode_batch("maple story"), top_k=1), [4])

        self.index.save() #serialize
        self.index = None

        self.index = MinHashIndex(self.tmp_dir.name, overwrite=False)

    def tearDown(self) -> None:
        pass

if __name__ == '__main__':
    unittest.main()
