import os
from typing import Dict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from main import *
from models.api import NNHashEncoder
import models
import random
from pprint import pprint
from xxhash import xxh32
import pandas as pd
import io
import numpy as np

def W2V_embedding_projector():
    import io
    from dataprocess.parser import XmlParser
    vecsTsvPath = os.path.join(os.path.dirname(HParams.filePath), "vecs-w2v.tsv")
    metaTsvPath = os.path.join(os.path.dirname(HParams.filePath), "meta-w2v.tsv")
    out_vecs = io.open(vecsTsvPath, 'w', encoding='utf-8')
    out_meta = io.open(metaTsvPath, 'w', encoding='utf-8')
    featureExtractor = HParams.getFeatureExtractor()
    out_meta.write(f"sentence\tencoding\tlength\n")
    xmlParser = XmlParser(HParams.filePath)
    sentences = (xmlParser.getSentsGenerator())()
    for sentence in sentences:
        clean = cleanQuery(sentence)
        batch_encoding = featureExtractor.get_feature_batch(clean)
        encoding = batch_encoding.reshape((16 * 200,))
        out_meta.write(f"{sentence}\t{encoding}\t{len(clean)}\n")
        out_vecs.write('\t'.join([str(x) for x in encoding]) + "\n")

    out_vecs.close()
    out_meta.close()


class ResultFactory(object):

    def __init__(self, use_default_ds_hash=False, hash_override=None,
                 model_type=None, train_range=None,
                 jaccard_threshold=0.0005, debug_hash_function=False):
        self.train_range = train_range if train_range else HParams.TRAIN_DATASET_RANGE
        self.model_type = model_type if model_type else HParams.MODEL_TYPE
        self.jaccard_threshold = jaccard_threshold
        self.encoder = self.get_hash_encoder()
        self.debug_hash_function = debug_hash_function
        if hash_override:
            self.hash = hash_override
        elif use_default_ds_hash:
            self.hash = ResultFactory.default_datasketch_hash
        else:
            self.hash = self.trained_model_hash

        self.index_path = os.path.join(os.path.dirname(HParams.filePath), f"index")
        self.index = None

    def get_hash_encoder(self):
        return models.api.getNNHashEncoder_New(
            restore_last=True, model_type=self.model_type, train_range=self.train_range
        )

    def autoencoder_vecs_save_meta(self, on_train_data=True, parse_range=HParams.PARSE_RANGE, hashfunc=None):
        xml_parser = XmlParser(HParams.filePath, trainDs=on_train_data, parseRange=parse_range)
        hasher = self.get_hash_encoder()
        vecs_tsv_path = os.path.join(os.path.dirname(HParams.filePath),
                                     f"autoencoder-vecs-{self.model_type}-{HParams.OUTPUT_DIM}.tsv")
        meta_tsv_path = os.path.join(os.path.dirname(HParams.filePath),
                                     f"autoencoder-meta-{self.model_type}-{HParams.OUTPUT_DIM}.tsv")
        out_vecs = io.open(vecs_tsv_path, 'w', encoding='utf-8')
        out_meta = io.open(meta_tsv_path, 'w', encoding='utf-8')
        out_meta.write(f"ID\tTitle\tEncoding\n")
        for post in tqdm.tqdm(xml_parser, total=parse_range[1]-parse_range[0]):
            wordsArr = post.toWordsArray()
            if len(wordsArr) == 0:
                continue
            if hashfunc is None:
                encoded_vecs = hasher.encode_batch(wordsArr)
            else:
                encoded_vecs = np.array(struct.unpack('ff', hashfunc((' '.join(wordsArr)).encode('utf-8')).digest()[:8]),
                                        dtype=np.float32)
            out_meta.write(f"{post.id}\t{post.title}\t{encoded_vecs}\n")
            out_vecs.write('\t'.join([str(x) for x in encoded_vecs]) + "\n")
        out_vecs.close()
        out_meta.close()

    def fill_and_save_index(self, index_path=None, jaccard_threshold=None, on_train_data=True,
                            parse_range=None, pass_as_str=True, num_perm=128):
        """
        :param parse_range: passed to the xml parser
        :param index_path: optional index save path
        :param jaccard_threshold: Jaccard similarity thershold for the LSH Minhash queries
        :return:
        """
        parse_range = parse_range or HParams.PARSE_RANGE
        xml_parser = XmlParser(HParams.filePath, trainDs=on_train_data, parseRange=parse_range, cachePostTitles=True)
        index_path = index_path or self.index_path
        index = NewMinHashIndex(index_path, overwrite=True, threshold=jaccard_threshold or self.jaccard_threshold,
                                hash_func=self.hash, pass_as_str=pass_as_str, num_perm=num_perm)
        for post in tqdm.tqdm(xml_parser, total=parse_range[1] - parse_range[0]):
            words_arr = post.toWordsArray()
            if len(words_arr) == 0:
                continue
            if pass_as_str:
                index.insert(post.id, ' '.join(words_arr))
            else:
                index.insert(post.id, words_arr)
        # index.save()
        self.index = index
        return index

    # def load_index(self, index_path=None):
    #     index_path = index_path or self.index_path
    #     self.index = NewMinHashIndex(index_path, overwrite=False)
    #     return self.index

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
        """
        encoded_vecs = self.encoder.encode_batch(str(data).split())
        encoded_vecs_bytes = encoded_vecs.tobytes()
        if self.debug_hash_function:
            print('-----------------------------------------------------------')
            print('----data', str(data))
            print('----encoded vec', encoded_vecs)
            print('----bytes', encoded_vecs_bytes)
        return struct.unpack('<I', encoded_vecs_bytes[:4])[0]



def fetch_post_by_id(id: str):
    xml_parser_train = XmlParser(HParams.filePath, trainDs=True, parseRange=HParams.PARSE_RANGE)
    post = [_ for _ in xml_parser_train if _.id == id]
    if len(post):
        return post[0]
    xml_parser_test = XmlParser(HParams.filePath, trainDs=False, parseRange=HParams.PARSE_RANGE)
    post = [_ for _ in xml_parser_test if _.id == id]
    if len(post):
        return post[0]
    return None


def compare_searches(search_res_include_titles=False, on_train_data=True, to_drop=1, parseRange=None,
                     **named_indexes):
    """
    :param search_res_include_titles: if True, the result of search will include also the corresponding title.
    :param on_train_data: passed to XmlParser as trainDs
    :param to_drop: amount of words to drop in second search
    :param named_indexes: passed minhash indexes
    :return: a dict in the format {title: {named_index: index_search(title)}}
    """
    parseRange = parseRange or HParams.PARSE_RANGE
    xml_parser = XmlParser(HParams.filePath, trainDs=on_train_data, parseRange=parseRange)
    res = {}
    for post in tqdm.tqdm(xml_parser, total=parseRange[1] - parseRange[0], desc="xml_parser"):
        words_arr = post.toWordsArray()
        if len(words_arr) == 0:
            continue

        # queries
        title = ' '.join(words_arr)  # as str
        as_str_queries = [title]
        as_lst_queries = [words_arr.copy()]
        for _ in range(to_drop):
            words_arr.pop(random.randrange(len(words_arr)))
        changed_title = ' '.join(words_arr)
        as_str_queries.append(changed_title)
        as_lst_queries.append(words_arr)
        # calc search results and fill
        for index_name, arg_index in named_indexes.items():
            if not arg_index.pass_as_str:
                queries = as_lst_queries
            else:
                queries = as_str_queries
            for q in queries:
                tmp = res.get((post.id, ' '.join(q)), {}) if isinstance(q, list) else res.get((post.id, q), {})
                if not search_res_include_titles:
                    tmp.update({index_name: arg_index.search(q)})
                else:
                    tmp.update({index_name: [(id, fetch_post_by_id(id).title) for id in arg_index.search(q)]})
                res[(post.id, ' '.join(q)) if isinstance(q, list) else (post.id, q)] = tmp

    return res


def results_dict_as_df(data_dict):
    df_dict = {'post_id': [], 'post_title': []}
    for (post_id, title), by_index_res in data_dict.items():
        df_dict['post_id'].append(post_id)
        df_dict['post_title'].append(title)
        for index_name, index_search_res in by_index_res.items():
            if index_name in df_dict:
                df_dict[index_name].append(post_id in index_search_res)
            else:
                df_dict[index_name] = [post_id in index_search_res]
    # df_dict[index_name] = tmp.append(post_id in index_search_res)
    result_df = pd.DataFrame(data=df_dict)
    return result_df


def save_meta(hashfunc=None):

    # fcn = ResultFactory(use_default_ds_hash=False, model_type='FCN')
    # fcn_2d.autoencoder_vecs_save_meta()
    cnn = ResultFactory(use_default_ds_hash=False, model_type='CNN')
    cnn.autoencoder_vecs_save_meta(hashfunc=hashfunc)




def main():
    # usage examples
    # HParams.MODEL_TYPE = "CNN"
    # HParams.TRAIN_DATASET_RANGE = (0, 1000)
    # HParams.MODEL_TYPE = "FCN"

    # save_meta()
    # return None, None

    # with default datasketch index hash
    with_default_hash = ResultFactory(use_default_ds_hash=True, jaccard_threshold=0.5)
    index_1 = with_default_hash.fill_and_save_index(pass_as_str=False)

    # with latest trained auto-encoder based hash
    fcn_hash = ResultFactory(use_default_ds_hash=False, jaccard_threshold=0.5, model_type='FCN')
    index_2 = fcn_hash.fill_and_save_index(pass_as_str=True)

    # with trained autoencoder from trained_weights_path based hash and jaccard similarity threshold set to 0.8
    cnn_hash = ResultFactory(use_default_ds_hash=False, jaccard_threshold=0.5, model_type='CNN')
    index_3 = cnn_hash.fill_and_save_index(on_train_data=True, pass_as_str=True)

    # with xxhash library based hash and jaccard similarity threshold set to 0.5
    xxhash_index = ResultFactory(hash_override=ResultFactory.xxhash, jaccard_threshold=0.5)
    index_4 = xxhash_index.fill_and_save_index(on_train_data=True, pass_as_str=False)

    # with sha3 based hash and jaccard similarity threshold set to 0.5
    sha3_hash_index = ResultFactory(hash_override=ResultFactory.sha3_hash, jaccard_threshold=0.5)
    index_5 = sha3_hash_index.fill_and_save_index(on_train_data=True, pass_as_str=False)

    # compare_searches takes Minhash index objects as named arguments and runs searches from all indexes on
    # either trained data or test. on each post the real title is queried along with a manipulated title with
    # to_drop dropped words
    indexes = dict(default_hash_index=index_1, fcn_hash=index_2, cnn_hash=index_3,
                   xxhash_index=index_4, sha3_hash_index=index_5)
    results_dict = compare_searches(search_res_include_titles=False, on_train_data=True, **indexes)
    # pprint(results_dict)

    df = results_dict_as_df(results_dict)
    agg_df = df.agg({index_name: ['sum'] for index_name in indexes.keys()})
    return df, agg_df


if __name__ == '__main__':
    df, agg_df = main()
    print(df)
    print(agg_df)
