# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from main import *
from models.DabaCnnAutoencoder import DabaCnnAutoencoder
from models.SimpleCnnAutoencoder import SimpleCnnAutoencoder
from models.SimpleFCNAutoencoder import SimpleFCNAutoencoder
from models.YabaDabaDiscriminator import DabaDiscriminator
from models.api import NNHashEncoder
import random
from pprint import pprint
from xxhash import xxh32


def autoencoder_vecs_save_meta(indexType=NewMinHashIndex, indexPath=None):
    import io
    xmlParser = XmlParser(HParams.filePath)
    indexPath = indexPath or os.path.join(os.path.dirname(HParams.filePath), "index")
    index = indexType(indexPath, overwrite=True)
    hasher = getNNHashEncoder()
    vecsTsvPath = os.path.join(os.path.dirname(HParams.filePath), "autoencoder-vecs.tsv")
    metaTsvPath = os.path.join(os.path.dirname(HParams.filePath), "autoencoder-meta.tsv")
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
    index.save()
    out_vecs.close()
    out_meta.close()
    return index


def W2V_embedding_projector():
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


class ResultFactory(object):

    def __init__(self, use_default_ds_hash=False, hash_override=None, model_type=HParams.MODEL_TYPE, jaccard_threshold=0.5,
                 debug_hash_function=False, trained_weights_path=None):
        self.trained_weights_path = trained_weights_path
        self.jaccard_threshold = jaccard_threshold
        self.model_type = model_type
        self.encoder = self.get_hash_encoder()
        self.debug_hash_function = debug_hash_function
        if hash_override:
            self.hash = hash_override
        elif use_default_ds_hash:
            self.hash = ResultFactory.default_datasketch_hash
        else:
            self.hash = self.trained_model_hash

        self.index_path = os.path.join(os.path.dirname(HParams.filePath), "index")
        self.index = None

    def get_hash_encoder(self):
        feature_extractor = HParams.getFeatureExtractor()
        models = {
            'DABA': DabaCnnAutoencoder(feature_extractor.get_feature_dim(), HParams.OUTPUT_DIM),
            'CNN': SimpleCnnAutoencoder(feature_extractor.get_feature_dim(), HParams.OUTPUT_DIM),
            'FCN': SimpleFCNAutoencoder(feature_extractor.get_feature_dim(), HParams.OUTPUT_DIM),
        }
        model = models[self.model_type]
        if HParams.MODEL_MODE == 'GAN':
            discriminator = DabaDiscriminator()
        else:
            discriminator = None

        return NNHashEncoder(model, discriminator, feature_extractor, restore_last=True,
                             chkp_path=self.trained_weights_path)

    def fill_and_save_index(self, index_path=None, jaccard_threshold=None, on_train_data=True):
        """
        :param index_path: optional index save path
        :param jaccard_threshold: Jaccard similarity thershold for the LSH Minhash queries
        :return:
        """
        xml_parser = XmlParser(HParams.filePath, trainDs=on_train_data)
        index_path = index_path or self.index_path
        index = NewMinHashIndex(index_path, overwrite=True, threshold=jaccard_threshold or self.jaccard_threshold,
                                hash_func=self.hash)
        for post in tqdm.tqdm(xml_parser):
            words_arr = post.toWordsArray()
            if len(words_arr) == 0:
                continue
            index.insert(post.id, ' '.join(words_arr))
        index.save()
        self.index = index
        return index

    def load_index(self, index_path=None):
        index_path = index_path or self.index_path
        self.index = NewMinHashIndex(index_path, overwrite=False)
        return self.index


    @staticmethod
    def xxhash(data):
        """ xxhash library based hash function """
        x = xxh32()
        x.update(data)
        return struct.unpack('<I', x.digest()[:4])[0]

    @staticmethod
    def sha3_hash(data):
        """ sha_3 based hash """
        return struct.unpack('<I', hashlib.sha3_224(data).digest()[:4])[0]

    @staticmethod
    def default_datasketch_hash(data):
        """ data sketch default hash for reference """
        return struct.unpack('<I', hashlib.sha1(data).digest()[:4])[0]  # original

    def trained_model_hash(self, data):
        """ 
        :param data: bytes to calculate hash on
        :return: hash using trained auto-encoder
        """""
        encoded_vecs = self.encoder.encode_batch(str(data).split())
        encoded_vecs_bytes = encoded_vecs.tobytes()
        if self.debug_hash_function:
            print('-----------------------------------------------------------')
            print('----data', str(data))
            print('----encoded vec', encoded_vecs)
            print('----bytes', encoded_vecs_bytes)
        return struct.unpack('<I', encoded_vecs_bytes[:4])[0]


def fetch_post_by_id(id: str):
    xml_parser_train = XmlParser(HParams.filePath, trainDs=True)
    post = [_ for _ in xml_parser_train if _.id == id]
    if len(post):
        return post[0]
    xml_parser_test = XmlParser(HParams.filePath, trainDs=False)
    post = [_ for _ in xml_parser_test if _.id == id]
    if len(post):
        return post[0]
    return None


def compare_searches(search_res_include_titles=False, on_train_data=True, to_drop=1, **named_indexes):
    """
    :param on_train_data: passed to XmlParser as trainDs
    :param to_drop: amount of words to drop in second search
    :param named_indexes: passed minhash indexes
    :return: a dict in the format {title: {named_index: index_search(title)}}
    """
    xml_parser = XmlParser(HParams.filePath, trainDs=on_train_data)
    res = {}
    for post in xml_parser:
        words_arr = post.toWordsArray()
        if len(words_arr) == 0:
            continue

        #queries
        title = ' '.join(words_arr)
        queries = [title]
        for _ in range(to_drop):
            words_arr.pop(random.randrange(len(words_arr)))
        changed_title = ' '.join(words_arr)
        queries.append(changed_title)

        #calc search results and fill
        for i, arg_index in named_indexes.items():
            for q in queries:
                tmp = res.get(q, {})
                if not search_res_include_titles:
                    tmp.update({i: arg_index.search(q)})
                else:
                    tmp.update({i: [(id, fetch_post_by_id(id).title) for id in arg_index.search(q)]})
                res[q] = tmp

    pprint(res)
    return res


if __name__ == '__main__':
    # usage examples

    # with default datasketch index hash
    with_default_hash = ResultFactory(use_default_ds_hash=True,
                                      trained_weights_path="checkpoints/train_SimpleCnnAutoencoder_1")
    index_1 = with_default_hash.fill_and_save_index()

    # with latest trained auto-encoder based hash
    with_our_hash = ResultFactory(use_default_ds_hash=False)
    index_2 = with_our_hash.fill_and_save_index()

    # with trained autoencoder from trained_weights_path based hash and jaccard similarity threshold set to 0.8
    additional_index = ResultFactory(use_default_ds_hash=False, jaccard_threshold=0.8,
                                     trained_weights_path="checkpoints/train_SimpleCnnAutoencoder_1")
    index_3 = additional_index.fill_and_save_index(on_train_data=True)  # note that on_train_data passed to trainDs in XmlParser

    # with xxhash library based hash and jaccard similarity threshold set to 0.5
    xxhash_index = ResultFactory(hash_override=ResultFactory.xxhash, jaccard_threshold=0.5,
                                 trained_weights_path="checkpoints/train_SimpleCnnAutoencoder_1")
    index_4 = xxhash_index.fill_and_save_index(on_train_data=True)

    # with sha3 based hash and jaccard similarity threshold set to 0.5
    sha3_hash_index = ResultFactory(hash_override=ResultFactory.sha3_hash, jaccard_threshold=0.5,
                                 trained_weights_path="checkpoints/train_SimpleCnnAutoencoder_1")
    index_5 = sha3_hash_index.fill_and_save_index(on_train_data=True)

    # compare_searches takes Minhash index objects as named arguments and runs searches from all indexes on
    # either trained data or test. on each post the real title is queried along with a manipulated title with
    # to_drop dropped words
    results_dict = compare_searches(search_res_include_titles=False, on_train_data=True, default_hash_index=index_1, our_hash_index=index_2, additional_index=index_3,
                                    xxhash_index=index_4, sha3_hash_index=index_5)

    # To print also the titles corresponding to returned ids in result
    # results_dict = compare_searches(search_res_include_titles=True, on_train_data=True, default_hash_index=index_1,
    #                                 our_hash_index=index_2)

    """
     sampled output of compare_searches:
    -----------------
    Note how our hash in this case performed better with the manipulated title in this examples:
    'subsonic nhibernate': {'additional_index': ['6222', '6210', '1383', '9473'],
                         'default_hash_index': [],
                         'our_hash_index': ['6222', '6210', '1383', '9473']},
    'subsonic vs nhibernate': {'additional_index': ['1384'],
                            'default_hash_index': ['1384'],
                            'our_hash_index': ['1384']},
                            
    Also:
     'using mstest': {'additional_index': ['1383', '9473', '6222', '6210'],
                  'default_hash_index': [],
                  'our_hash_index': ['1383', '9473', '6222', '6210'],
                  'sha3_hash_index': [],
                  'xxhash_index': []},
     'using mstest cruisecontrolnet': {'additional_index': ['1314'],
                                   'default_hash_index': ['1314'],
                                   'our_hash_index': ['1314'],
                                   'sha3_hash_index': ['1314'],
                                   'xxhash_index': ['1314']},
    """







