from dataprocess.parser import XmlParser
from hparams import HParams
from index.hash_index import Index
from index.index import MinHashIndex
from models.api import getNNHashEncoder
from train_and_test import encode, encode_batch
import numpy as np
import os
from dataprocess.cleaners import cleanQuery
import tensorflow as tf

def train_partial(*args, **kwargs):
    import models.train
    models.train.train_yabadaba(*args, **kwargs, dataset_type="partial_titles")


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
    index.save()
    return index

def saveYabaDabaIndex():
    xmlParser = XmlParser(HParams.filePath)
    indexPath = os.path.join(os.path.dirname(HParams.filePath), "index")
    index = MinHashIndex(indexPath, overwrite=True)
    hasher = getNNHashEncoder()
    for post in xmlParser:
        wordsArr = post.toWordsArray()
        encodedVecs = hasher.encode_batch(wordsArr)
        index.insert(post.id, encodedVecs)
    index.index()
    index.save()
    return index

def runSearch(index, searchQuery = None):
    wordsArr = cleanQuery(searchQuery)
    hasher = getNNHashEncoder()
    encodedVecs = hasher.encode_batch(wordsArr)
    print(index.search(encodedVecs, top_k=10))

if __name__ == '__main__':
    # train_example(epochs=1000, restore_last=False, progress_per_step=100)
    train_partial(epochs=10, restore_last=False, progress_per_step=10)
    index = saveYabaDabaIndex()
    runSearch(index, "What are the advantages of using SVN over CVS")
    runSearch(index, "ASP.Net Custom Client-Side Validation")
