import unittest
import logging
from index.index import MinHashIndex
from tempfile import TemporaryDirectory
import numpy as np


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.index = MinHashIndex(self.tmp_dir.name, overwrite=True)

    def test_search_and_insert(self):
        self.index.insert(1, [1, 1, 1, 0, 1, 0, 0, 1, 0, 0])
        self.index.insert(2, [1, 0, 1, 0, 1, 0, 0, 1, 0, 0])
        self.index.insert(3, [1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
        self.index.insert(4, [1, 0, 1, 1, 1, 0, 0, 1, 1, 1])

        self.index.index()
        self.assertEqual(self.index.search([1, 1, 1, 0, 1, 0, 0, 1, 0, 0], top_k=1), [1])
        self.assertEqual(self.index.search([1, 0, 1, 0, 1, 1, 0, 1, 0, 1], top_k=1), [3])
        self.assertEqual(self.index.search([1, 0, 1, 0, 1, 0, 0, 1, 0, 0], top_k=1), [2])
        self.assertEqual(self.index.search([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], top_k=4), [1, 2, 3, 4])

        self.index.save() #serialize
        self.index = None

        self.index = MinHashIndex(self.tmp_dir.name, overwrite=False)
        self.assertEqual(self.index.search([1, 1, 1, 0, 1, 0, 0, 1, 0, 0], top_k=1), [1])
        self.assertEqual(self.index.search([1, 0, 1, 0, 1, 1, 0, 1, 0, 1], top_k=1), [3])
        self.assertEqual(self.index.search([1, 0, 1, 0, 1, 0, 0, 1, 0, 0], top_k=1), [2])
        self.assertEqual(self.index.search([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], top_k=4), [1, 2, 3, 4])

    def tearDown(self) -> None:
        pass

if __name__ == '__main__':
    unittest.main()
