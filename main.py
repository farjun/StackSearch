# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
import tqdm
import models.train
from dataprocess.cleaners import cleanQuery
from dataprocess.parser import XmlParser
from hparams import HParams
from index.index import MinHashIndex
from index.index_new import MinHashIndex as NewMinHashIndex
from models.api import getNNHashEncoder


def train_partial(*args, **kwargs):
    import models.train
    models.train.train_and_test_yabadaba(*args, **kwargs, dataset_type="partial_titles")


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


def runSearch(index, searchQuery=None, returnEncoded=False, shouldBe=None):
    print('=' * 10)
    print(searchQuery)
    print('=' * 10)
    wordsArr = cleanQuery(searchQuery)
    hasher = getNNHashEncoder()
    encodedVecs = hasher.encode_batch(wordsArr)
    print(encodedVecs)
    res = index.search(encodedVecs, top_k=10)
    if returnEncoded:
        return res, encodedVecs
    if shouldBe:
        return "{shouldBe} in {res} = {isIn}".format(res=res, shouldBe=shouldBe, isIn=str(shouldBe) in res)
    return res


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


def W2VembeddingProjector():
    import io
    from dataprocess.parser import XmlParser
    xmlParser = XmlParser(HParams.filePath)
    vecsTsvPath = os.path.join(os.path.dirname(HParams.filePath), "vecs-w2v.tsv")
    metaTsvPath = os.path.join(os.path.dirname(HParams.filePath), "meta-w2v.tsv")
    out_vecs = io.open(vecsTsvPath, 'w', encoding='utf-8')
    out_meta = io.open(metaTsvPath, 'w', encoding='utf-8')
    featureExtractor = HParams.getFeatureExtractor()
    out_meta.write(f"sentence\tencoding\tlength\n")
    xmlParser = XmlParser(HParams.filePath)
    sentences=(xmlParser.getSentsGenerator())()
    for sentence in sentences:
        clean = cleanQuery(sentence)
        batch_encoding=featureExtractor.get_feature_batch(clean)
        encoding=batch_encoding.reshape((16*200,))
        out_meta.write(f"{sentence}\t{encoding}\t{len(clean)}\n")
        out_vecs.write('\t'.join([str(x) for x in encoding]) + "\n")

    out_vecs.close()
    out_meta.close()


def main(**kwargs):
    """
    :param kwargs: args which will be passed to train_partial -> train_yabadaba
    """
    models.train.train_and_test_yabadaba(**kwargs)


def clear_summary():
    import shutil
    shutil.rmtree("summary", ignore_errors=True)


def generate_w2v(*args, **kwargs):
    import models.train
    models.train.train_embedding_word2vec_new(*args, **kwargs)


if __name__ == '__main__':
    # generate_w2v()
    # clear_summary()
    main(epochs=1, restore_last=False, progress_per_step=2)
    # indexPath = os.path.join(os.path.dirname(HParams.filePath), "index")
    # xmlParser = XmlParser(HParams.filePath)
    # index = saveYabaDabaIndex()
    indexPath = os.path.join(os.path.dirname(HParams.filePath), "index")
    index = MinHashIndex(indexPath)
    print("index size: ".format(index.size()))

    print(runSearch(index, "Determine a user's timezone", shouldBe=13))  # should be 13
    print(runSearch(index, "Converting ARBG to RGB alpha blending", shouldBe=2780))  # should be 2780
    # [0.9987422  0.9987696  0.99872804 0.9987277 ]
    print(runSearch(index, "Regex: To pull out a sub-string between two tags in a string",
                    shouldBe=1237))  # should be 1237
    print(runSearch(index, "ASP.Net Custom Client-Side Validation", shouldBe=1401))  # should be 1401
#     runSearches([
#         "What are the advantages of using SVN over CVS",
#         "ASP.Net Custom Client-Side Validation",
#         "regex to pull"
#     ])

    index_ = NewMinHashIndex(indexPath, overwrite=True, threshold=0.3)
    index_.insert(4, ['luke', 'dunphy'])
    index_.insert(5, ['phill', 'dunphy'])
    index_.insert(6, ['haley', 'dunphy'])
    print(index_.search(['alex', 'dunphy']))

    # index_.insert(1, ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
    #         'estimating', 'the', 'similarity', 'between', 'datasets'])
    index_.insert(2, ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
                        'estimating', 'the', 'similarity', 'between', 'documents'])
    index_.insert(3, ['minhash', 'is', 'probability', 'data', 'structure', 'for',
                        'estimating', 'the', 'similarity', 'between', 'documents'])
    print(index_.search(['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
                         'estimating', 'the', 'similarity', 'between', 'datasets']))
