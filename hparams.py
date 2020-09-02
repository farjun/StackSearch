from os import path

class HParams:
    filePath = path.join("data", "partial", "Posts.xml")
    word2vecFilePath = path.join("checkpoints", "word2vec", "w2v_embedding")
    doc2vecFilePath = path.join("checkpoints", "word2vec", "doc_embedding")
    BATCH_SIZE = 4
    OUTPUT_DIM = 64
    CKPT_MAX_TO_KEEP = 2
    MAX_SENTENCE_DIM = 32

    def getCeckpointPath(self):
        return str(self.OUTPUT_DIM)




