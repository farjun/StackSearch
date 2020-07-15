import unittest
import logging
from index.hash_index import Index
from tempfile import TemporaryDirectory


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.index = Index(self.tmp_dir.name, disk_chunk_size=2)

    def test_search_and_insert(self):
        self.index.insert([1, 1, 1, 0], "http://stack/omer")
        self.index.insert([1, 0, 1, 0], "http://stack.ori")
        self.index.insert([1, 1, 1, 1], "http://dump.point")
        self.index.insert([1, 0, 0, 0], "http://dump.point")
        self.index.insert([1, 0, 0, 1], "http://dump.point")
        self.index.insert([1, 1, 0, 1], "http://dump.point")
        self.index.insert([0, 1, 0, 1], "http://dump.point")

        self.index.sort()
        self.index.print()
        self.assertEqual(len(self.index.brute_force_search([1, 0, 1, 1], dist_limit=0)), 0)
        self.assertEqual(len(self.index.brute_force_search([1, 0, 1, 0], dist_limit=0)), 1)
        print(self.index.brute_force_search([1, 0, 1, 0], dist_limit=0))
        print('-'*10)
        print(list(self.index.search([1, 0, 1, 0], result_size_limit=3)))
        self.assertLessEqual(len(self.index.search([1, 0, 1, 0], result_size_limit=3)), 3)  # binary search

if __name__ == '__main__':
    unittest.main()
