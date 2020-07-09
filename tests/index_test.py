import unittest
from index.hash_index import Index
from tempfile import TemporaryDirectory


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.index = Index(self.tmp_dir.name, threshold=1)

    def test_search_and_insert(self):
        self.index.insert([1, 1, 1, 0], "http://stack/omer")
        self.index.insert([1, 0, 1, 0], "http://stack.ori")
        self.index.insert([1, 1, 1, 1], "http://dump.point")
        self.assertEqual(len(self.index.search([1, 1, 1, 1], dist_limit=0.5)), 2)
        self.assertEqual(len(self.index.search([1, 0, 1, 1], dist_limit=0)), 0)
        self.assertEqual(len(self.index.search([1, 0, 1, 0], dist_limit=0)), 1)



if __name__ == '__main__':
    unittest.main()
