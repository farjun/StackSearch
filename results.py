# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from main import *
from models.DabaCnnAutoencoder import DabaCnnAutoencoder
from models.SimpleCnnAutoencoder import SimpleCnnAutoencoder
from models.SimpleFCNAutoencoder import SimpleFCNAutoencoder
from models.YabaDabaDiscriminator import DabaDiscriminator
from models.api import NNHashEncoder


def autoencoder_vecs_save_meta(indexType=NewMinHashIndex, indexPath=None, calc_and_save_encodings=True ):
    import io
    xmlParser = XmlParser(HParams.filePath)
    indexPath = indexPath or os.path.join(os.path.dirname(HParams.filePath), "index")
    index = indexType(indexPath, overwrite=True)
    if calc_and_save_encodings:
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
        return index

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


if __name__ == '__main__':
    # main(epochs=1, restore_last=True, progress_per_step=10)
    factory = ResultFactory()
    index = factory.fill_and_save_index()
    print(f"index size: {index.size()}")

    index_loaded = ResultFactory().load_index()
    print(f"loaded index size: {index_loaded.size()}")





