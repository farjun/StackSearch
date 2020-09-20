from os import path


class HParams:
    USE_CNN = False
    LR = 1e-4
    USE_REGULARIZER = True
    USE_BATCH_NORMALIZATION = False
    REGULARIZER_L2 = 0.1
    REGULARIZER_L1 = 0.1
    DATASET = 'titles'
    TRAIN_DATASET_SIZE = 1000  # takes first DATASET_SIZE posts out of the DATASET
    TEST_DATASET_SIZE = 100  # takes first DATASET_SIZE posts out of the DATASET
    CROSS_ENTROPY_LOSS_LAMBDA = 1
    RECONSTRUCTION_LOSS_LAMBDA = 1
    filePath = path.join("data", "Posts.xml")
    AMOUNT_TO_DROP = 3
    AMOUNT_TO_SWAP = 3
    embeddingFilePath = path.join("checkpoints", "word2vec")
    BATCH_SIZE = 4
    OUTPUT_DIM = 4
    CKPT_MAX_TO_KEEP = 2
    MAX_SENTENCE_DIM = 16

    def getCeckpointPath(self):
        return str(self.OUTPUT_DIM)

    @staticmethod
    def getFeatureExtractor(**kwargs):
        from features.FeatureExtractors import WordEmbeddingToMatrixFeatureExtractor, D2VFeatureExtractor, \
            W2VFeatureExtractor, FeatureExtractor_Temp
        return FeatureExtractor_Temp(**kwargs)

    @staticmethod
    def getFeatureExtractorDim():
        return HParams.getFeatureExtractor().get_feature_dim()
