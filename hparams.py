from os import path

class HParams:
    DATASET = 'titles'
    DATASET_SIZE = 20 # takes first DATASET_SIZE posts out of the DATASET
    filePath = path.join("data", "Posts.xml")

    embeddingFilePath = path.join("checkpoints", "word2vec")
    BATCH_SIZE = 4
    OUTPUT_DIM = 64
    CKPT_MAX_TO_KEEP = 2
    MAX_SENTENCE_DIM = 32


    def getCeckpointPath(self):
        return str(self.OUTPUT_DIM)

    @staticmethod
    def getFeatureExtractor(**kwargs):
        from features.FeatureExtractors import WordEmbeddingToMatrixFeatureExtractor, D2VFeatureExtractor, \
            W2VFeatureExtractor
        return W2VFeatureExtractor(**kwargs)

    @staticmethod
    def getFeatureExtractorDim():
        return HParams.getFeatureExtractor().get_feature_dim()