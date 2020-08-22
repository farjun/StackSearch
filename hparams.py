from os import path

class HParams:
    DATASET_SIZE = 20000 # takes first DATASET_SIZE posts out of the full dataset
    filePath = path.join("data", "Posts.xml")
    BATCH_SIZE = 4
    OUTPUT_DIM = 64
    CKPT_MAX_TO_KEEP = 2
    MAX_SENTENCE_DIM = 32

    def getCeckpointPath(self):
        return str(self.OUTPUT_DIM)




