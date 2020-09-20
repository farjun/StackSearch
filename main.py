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

def test_yabadaba(*args, **kwargs):
    import models.train
    models.train.test_yabadaba(*args, **kwargs, dataset_type="partial_titles")

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


def saveYabaDabaIndexWithMeta(saveIndexPath=None):
    import io
    xmlParser = XmlParser(HParams.filePath)
    indexPath = saveIndexPath or os.path.join(os.path.dirname(HParams.filePath), "index")
    index = MinHashIndex(indexPath, overwrite=True)
    hasher = getNNHashEncoder()
    vecsTsvPath = os.path.join(os.path.dirname(HParams.filePath), "vecs.tsv")
    metaTsvPath = os.path.join(os.path.dirname(HParams.filePath), "meta_1.tsv")
    out_vecs = io.open(vecsTsvPath, 'w', encoding='utf-8')
    out_meta = io.open(metaTsvPath, 'w', encoding='utf-8')
    out_meta.write(f"ID\tTitle\tEncoding\n")
    for post in tqdm.tqdm(xmlParser):
        wordsArr = post.toWordsArray()
        if len(wordsArr) == 0:
            continue
        encodedVecs = hasher.encode_batch(wordsArr)
        index.insert(post.id, encodedVecs)
        out_meta.write(f"{post.id}\t{post.title}\t{encodedVecs}\n")
        out_vecs.write('\t'.join([str(x) for x in encodedVecs]) + "\n")
    index.index()
    index.save()
    out_vecs.close()
    out_meta.close()
    return index

def runSearch(index, searchQuery=None, returnEncoded=False):
    print('=' * 10)
    print(searchQuery)
    print('=' * 10)
    wordsArr = cleanQuery(searchQuery)
    hasher = getNNHashEncoder()
    encodedVecs = hasher.encode_batch(wordsArr)
    print(encodedVecs)
    if returnEncoded:
        return index.search(encodedVecs, top_k=10), encodedVecs
    return index.search(encodedVecs, top_k=10)


def runSearches(searches: list):
    indexPath = os.path.join(os.path.dirname(HParams.filePath), "index")
    index = MinHashIndex(indexPath, overwrite=False)
    all_vecs = []
    for search in searches:
        no_mask = runSearch(index, search)
        all_vecs.append(no_mask)

def embeddingProjectorPrep(searches: list):
    import io
    indexPath = os.path.join(os.path.dirname(HParams.filePath), "index")
    vecsTsvPath = os.path.join(os.path.dirname(HParams.filePath), "vecs.tsv")
    metaTsvPath = os.path.join(os.path.dirname(HParams.filePath), "meta.tsv")
    out_vecs = io.open(vecsTsvPath, 'w', encoding='utf-8')
    out_meta = io.open(metaTsvPath, 'w', encoding='utf-8')
    index = MinHashIndex(indexPath, overwrite=False)

    out_meta.write(f"search\ttop_k\tencoded\n")
    for search in searches:
        top_k, encoded = runSearch(index, search, returnEncoded=True)
        out_meta.write(f"{search}\t{top_k}\t{encoded}\n")
        out_vecs.write('\t'.join([str(x) for x in encoded]) + "\n")

    out_vecs.close()
    out_meta.close()

def main(**kwargs):
    """
    :param kwargs: args which will be passed to train_partial -> train_yabadaba
    """
    # models.train.train_yabadaba(**kwargs)
    kwargs['restore_last'] = True
    models.train.test_yabadaba(**kwargs)

def clear_summary():
    import shutil
    shutil.rmtree(os.path.join("summary","train"),ignore_errors=True)


def generate_w2v(*args, **kwargs):
    import models.train
    models.train.train_embedding_word2vec_new(*args, **kwargs)


if __name__ == '__main__':
    # generate_w2v()

    main(epochs=5, restore_last=True, progress_per_step=2)
    # indexPath = os.path.join(os.path.dirname(HParams.filePath), "index")
    xmlParser = XmlParser(HParams.filePath)
    index = saveYabaDabaIndex()
    # indexPath = os.path.join(os.path.dirname(HParams.filePath), "index")
    # index = MinHashIndex(indexPath)
    # print("index size: ".format(index.size()))

    print(runSearch(index, "Determine a user's timezone")) # should be 13
    print(runSearch(index, "Converting ARBG to RGB alpha blending")) # should be 2780
    print(runSearch(index, "Regex: To pull out a sub-string between two tags in a string")) # should be 1237
    print(runSearch(index, "ASP.Net Custom Client-Side Validation")) # should be 1401
#     runSearches([
#         "What are the advantages of using SVN over CVS",
#         "ASP.Net Custom Client-Side Validation",
#         "regex to pull"
#     ])
