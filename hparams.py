from os import path

class HParams:
    MODEL_TYPE = 'FCN' # DABA, CNN, FCN
    MODEL_MODE = 'NOT_GAN' # GAN, NOT_GAN
    LR = 1e-4
    USE_REGULARIZER = True
    USE_BATCH_NORMALIZATION = False
    REGULARIZER_L2 = 0.1
    REGULARIZER_L1 = 0.1
    DATASET = 'titles'
    TRAIN_DATASET_RANGE = (0, 100000)
    TEST_DATASET_RANGE = (100000, 101000)
    PARSE_RANGE = (0, 50000)
    CROSS_ENTROPY_LOSS_LAMBDA = 1
    RECONSTRUCTION_LOSS_LAMBDA = 1
    filePath = path.join("data", "Posts.xml")
    AMOUNT_TO_DROP = 3
    MAX_AMOUNT_TO_DROP = 6
    AMOUNT_TO_SWAP = 3
    MAX_AMOUNT_TO_SWAP = 6
    embeddingFilePath = path.join("checkpoints", "word2vec")
    BATCH_SIZE = 4
    OUTPUT_DIM = 1
    CKPT_MAX_TO_KEEP = 2
    MAX_SENTENCE_DIM = 16
    LAST_ACTIVATION = None

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
