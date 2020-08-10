import unittest
import logging
from index.hash_index import Index
from tempfile import TemporaryDirectory


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.index = Index(self.tmp_dir.name, disk_chunk_size=2, to_erase_on_raised_exception=False)
        self.perm_index = Index("./index", disk_chunk_size=2, to_erase_on_raised_exception=True)

    def test_search_and_insert(self):
        self.index.insert([1, 1, 1, 0], "http://stack/omer")
        self.index.insert([1, 0, 1, 0], "http://stack.ori")
        self.index.insert([1, 1, 1, 1], "http://dump.point")
        self.index.insert([1, 0, 0, 0], "http://dump.point")
        self.index.insert([1, 0, 0, 1], "http://dump.point")
        self.index.insert([1, 1, 0, 1], "http://dump.point")
        self.index.insert([0, 1, 0, 1], "http://dump.point")

        self.perm_index.insert([0, 1, 0, 1], "http://dump.point")
        self.perm_index.sort()

        self.index.sort()
        self.index.print()
        self.assertEqual(len(self.index.brute_force_search([1, 0, 1, 1], dist_limit=0)), 0)
        print(self.index.brute_force_search([1, 0, 1, 0], dist_limit=0))
        print('-'*10)
        print(list(self.index.search([1, 0, 1, 0], result_size_limit=3)))
        self.assertLessEqual(len(self.index.search([1, 0, 1, 0], result_size_limit=3)), 3)  # binary search

    def tearDown(self) -> None:
        pass

if __name__ == '__main__':
    unittest.main()
