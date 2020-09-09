from dataprocess.parser import XmlParser
from hparams import HParams
from index.hash_index import Index
from index.index import MinHashIndex
from models.api import getNNHashEncoder
import numpy as np
import os
from dataprocess.cleaners import cleanQuery
import tensorflow as tf
import models.train

def saveYabaDabaIndex(saveIndexPath=None):
    xmlParser = XmlParser(HParams.filePath)
    indexPath = saveIndexPath or os.path.join(os.path.dirname(HParams.filePath), "index")
    index = MinHashIndex(indexPath, overwrite=True)
    hasher = getNNHashEncoder()
    for post in xmlParser:
        wordsArr = post.toWordsArray()
        encodedVecs = hasher.encode_batch(wordsArr)
        index.insert(post.id, encodedVecs)
    index.index()
    index.save()
    return index

def runSearch(index, searchQuery=None):
    print('='*10)
    print(searchQuery)
    print('='*10)
    wordsArr = cleanQuery(searchQuery)
    hasher = getNNHashEncoder()
    encodedVecs = hasher.encode_batch(wordsArr)
    print(encodedVecs)
    return index.search(encodedVecs, top_k=10)


def main(**kwargs):
    """
    :param kwargs: args which will be passed to train_partial -> train_yabadaba
    """
    models.train.train_yabadaba(**kwargs)
    indexPath = os.path.join(os.path.dirname(HParams.filePath), "index")
    index = MinHashIndex(indexPath, overwrite=False)
    if True or index.size() != HParams.DATASET_SIZE:
        print("HParams.DATASET_SIZE != index.size() : {} != {}, indexing again".format(HParams.DATASET_SIZE,
                                                                                       index.size()))
        index = saveYabaDabaIndex()
        print("index size: ".format(index.size()))

def clear_summary():
    import shutil
    shutil.rmtree(os.path.join("summary","train"),ignore_errors=True)


if __name__ == '__main__':
    main(epochs=100, restore_last=False, progress_per_step=2)
    indexPath = os.path.join(os.path.dirname(HParams.filePath), "index")
    index = MinHashIndex(indexPath, overwrite=False)
    print(runSearch(index, "Regex: To pull out a sub-string between two tags in a string"))
    print(runSearch(index, "ASP.Net Custom Client-Side Validation"))