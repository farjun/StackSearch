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
    print(encodedVecs)  # TODO check why encodedVecs are the same for diff sentences
    return index.search(encodedVecs, top_k=10)

def runSearches(searches:list):
    indexPath = os.path.join(os.path.dirname(HParams.filePath), "index")
    index = MinHashIndex(indexPath, overwrite=False)
    for search in searches:
        runSearch(index,search)

def main(**kwargs):
    """
    :param kwargs: args which will be passed to train_partial -> train_yabadaba
    """
    train_partial(**kwargs)
    indexPath = os.path.join(os.path.dirname(HParams.filePath), "index")
    index = MinHashIndex(indexPath, overwrite=False)
    if index.size( ) != HParams.DATASET_SIZE:
        print("HParams.DATASET_SIZE != index.size() : {} != {}, indexing again".format(HParams.DATASET_SIZE,
                                                                                       index.size()))
        index = saveYabaDabaIndex()
    # print(runSearch(index, "What are the advantages of using SVN over CVS"))
    # print(runSearch(index, "ASP.Net Custom Client-Side Validation"))
    # print(index.size())

def generate_w2v():
    import models.train
    models.train.train_embedding_word2vec_new()

if __name__ == '__main__':
    # generate_w2v()
    main(epochs=100, restore_last=False, progress_per_step=2)
    runSearches([
        "What are the advantages of using SVN over CVS",
        "ASP.Net Custom Client-Side Validation",
        "py pasjdasdj  aopsdk kdoas odkasldas dkasl; ;lsad;aldalkasdl;k asdasd asdas "
    ])