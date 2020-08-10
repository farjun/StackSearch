from dataprocess.parser import XmlParser
from hparams import HParams
from index.hash_index import Index
from index.index import MinHashIndex
from train_and_test import encode, encode_batch
import numpy as np
import os

def train_partial(*args, **kwargs):
    import models.train
    models.train.train_yabadaba(*args, **kwargs, dataset_type="partial")


def train_example(*args, **kwargs):
    import train_and_test
    train_and_test.train(*args, **kwargs, dataset_type="example")


def saveIndex():
    xmlParser = XmlParser(HParams.filePath)
    indexPath = os.path.join(os.path.dirname(HParams.filePath), "index")
    index = MinHashIndex(indexPath, overwrite=True)
    for post in xmlParser:
        wordsArr = post.toWordsArray()
        encodedVecs = encode_batch(wordsArr)
        postSimHash = np.around(np.average(encodedVecs, axis=0))
        index.insert(post.id, postSimHash)
    index.sort()
    index.save()
    return index

def runSearch(index):
    wordsArr = ["Python", "numpy"]
    encodedVecs = encode_batch(wordsArr)
    simHash = np.around(np.average(encodedVecs, axis=0))
    print(index.search(simHash))

if __name__ == '__main__':
    # train_example(epochs=1000, restore_last=False, progress_per_step=100)
    train_partial(epochs=10, restore_last=True, progress_per_step=10)
    index = saveIndex()
    runSearch(index)
