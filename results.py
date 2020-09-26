# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from main import *
from models.DabaCnnAutoencoder import DabaCnnAutoencoder
from models.SimpleCnnAutoencoder import SimpleCnnAutoencoder
from models.SimpleFCNAutoencoder import SimpleFCNAutoencoder
from models.YabaDabaDiscriminator import DabaDiscriminator
from models.api import NNHashEncoder
import random
from pprint import pprint

def autoencoder_vecs_save_meta(indexType=NewMinHashIndex, indexPath=None, calc_and_save_encodings=True ):
    import io
    xmlParser = XmlParser(HParams.filePath)
    indexPath = indexPath or os.path.join(os.path.dirname(HParams.filePath), "index")
    index = indexType(indexPath, overwrite=True)
    if calc_and_save_encodings:
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

    def __init__(self, use_default_ds_hash=True, model_type=HParams.MODEL_TYPE, jaccard_threshold=0.5,
                 debug_hash_function=False):
        self.jaccard_threshold = jaccard_threshold
        self.model_type = model_type
        self.encoder = self.get_hash_encoder()
        self.debug_hash_function = debug_hash_function
        if use_default_ds_hash:
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

        return NNHashEncoder(model, discriminator, feature_extractor, restore_last=True)

    def fill_and_save_index(self, index_path=None, jaccard_threshold=None):
        """
        :param index_path: optional index save path
        :param jaccard_threshold: Jaccard similarity thershold for the LSH Minhash queries
        :return:
        """
        xml_parser = XmlParser(HParams.filePath)
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


def compare_searches(*args):
    """
    :param args: passed minhash indexes
    :return:
    """
    xml_parser = XmlParser(HParams.filePath, trainDs=True)
    res = {}
    for post in tqdm.tqdm(xml_parser):
        words_arr = post.toWordsArray()
        if len(words_arr) == 0:
            continue
        print('-'*10)
        print('TITLE:', post.title)

        #queries
        title = ' '.join(words_arr)
        changed_title = ' '.join(words_arr.pop(random.randrange(len(words_arr))))
        queries = [title, changed_title]

        #calc search results and fill
        for i, arg_index in enumerate(args):
            for q in queries:
                tmp = res.get(q, {})
                tmp.update({i: arg_index.search(q)})
                res[q] = tmp

        pprint(res)


if __name__ == '__main__':
    # main(epochs=1, restore_last=True, progress_per_step=10)

    with_default_hash = ResultFactory(use_default_ds_hash=True, debug_hash_function=True)
    index_1 = with_default_hash.fill_and_save_index()

    with_our_hash = ResultFactory(use_default_ds_hash=False)
    index_2 = with_our_hash.fill_and_save_index()

    compare_searches(index_1, index_2)

    print(f"index 1 size: {index_1.size()}")
    print(f"index 2 size: {index_2.size()}")








