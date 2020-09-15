# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
import tqdm
import models.train
from dataprocess.cleaners import cleanQuery
from dataprocess.parser import XmlParser
from hparams import HParams
from index.index import MinHashIndex
from models.api import getNNHashEncoder


def train_partial(*args, **kwargs):
    import models.train
    models.train.train_yabadaba(*args, **kwargs, dataset_type="partial_titles")


def saveYabaDabaIndex(saveIndexPath=None):
    xmlParser = XmlParser(HParams.filePath)
    indexPath = saveIndexPath or os.path.join(os.path.dirname(HParams.filePath), "index")
    index = MinHashIndex(indexPath, overwrite=True)
    hasher = getNNHashEncoder()
    for post in tqdm.tqdm(xmlParser):
        wordsArr = post.toWordsArray()
        if len(wordsArr) == 0:
            continue
        encodedVecs = hasher.encode_batch(wordsArr)
        index.insert(post.id, encodedVecs)
    index.index()
    index.save()
    return index

def runSearch(index, searchQuery=None):
    print('=' * 10)
    print(searchQuery)
    print('=' * 10)
    wordsArr = cleanQuery(searchQuery)
    hasher = getNNHashEncoder()
    encodedVecs = hasher.encode_batch(wordsArr)
    print(encodedVecs)
    return index.search(encodedVecs, top_k=10)


def runSearches(searches: list):
    indexPath = os.path.join(os.path.dirname(HParams.filePath), "index")
    index = MinHashIndex(indexPath, overwrite=False)
    all_vecs = []
    for search in searches:
        no_mask = runSearch(index, search)
        all_vecs.append(no_mask)

def main(**kwargs):
    """
    :param kwargs: args which will be passed to train_partial -> train_yabadaba
    """
    models.train.train_yabadaba(**kwargs)


def clear_summary():
    import shutil
    shutil.rmtree(os.path.join("summary","train"),ignore_errors=True)


def generate_w2v(*args, **kwargs):
    import models.train
    models.train.train_embedding_word2vec_new(*args, **kwargs)


if __name__ == '__main__':
    # generate_w2v()
    main(epochs=100, restore_last=False, progress_per_step=2)
    # indexPath = os.path.join(os.path.dirname(HParams.filePath), "index")
    index = saveYabaDabaIndex()
    print("index size: ".format(index.size()))

    print(runSearch(index, "Determine a user's timezone")) # should be 13
    print(runSearch(index, "Regex: To pull out a sub-string between two tags in a string")) # should be 1237
    print(runSearch(index, "ASP.Net Custom Client-Side Validation")) # should be 1401
#     runSearches([
#         "What are the advantages of using SVN over CVS",
#         "ASP.Net Custom Client-Side Validation",
#         "regex to pull"
#     ])
