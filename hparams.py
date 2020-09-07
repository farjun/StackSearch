from os import path

class HParams:
    DATASET_SIZE = 2 # takes first DATASET_SIZE posts out of the full dataset
    filePath = path.join("data", "Posts.xml")
    embeddingFilePath = path.join("checkpoints", "word2vec")
    BATCH_SIZE = 4
    OUTPUT_DIM = 64
    CKPT_MAX_TO_KEEP = 2
    MAX_SENTENCE_DIM = 32
    featureExtractor = None
    def getCeckpointPath(self):
        return str(self.OUTPUT_DIM)

    @staticmethod
    def getFeatureExtractor(**kwargs):
        from features.FeatureExtractors import WordEmbeddingToMatrixFeatureExtractor, D2VFeatureExtractor, W2VFeatureExtractor
        return W2VFeatureExtractor(**kwargs)