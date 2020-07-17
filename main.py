from dataprocess.parser import XmlParser
from hparams import HParams
from index.hash_index import Index
from train_and_test import encode, encode_batch
import numpy as np

def train_partial(*args, **kwargs):
    import train_and_test
    train_and_test.train(*args, **kwargs, dataset_type="partial")


def train_example(*args, **kwargs):
    import train_and_test
    train_and_test.train(*args, **kwargs, dataset_type="example")


def saveIndex():
    xmlParser = XmlParser(HParams.filePath)
    indexPath = HParams.filePath.rsplit('\\', 1)[0] + "\\index\\"
    hashIndex = Index(indexPath)
    for post in xmlParser:
        wordsArr = post.toWordsArray()
        assert wordsArr[0] != wordsArr[1]
        encodedVecs = encode_batch(wordsArr)
        assert np.all(encodedVecs[0] != encodedVecs[1])
        postSimHash = np.average(encodedVecs, axis=0)
        hashIndex.insert(post.id, val=postSimHash)

if __name__ == '__main__':
    # train_example(epochs=1000, restore_last=False, progress_per_step=100)
    #train_partial(epochs=1000, restore_last=True, progress_per_step=100)
    saveIndex()
