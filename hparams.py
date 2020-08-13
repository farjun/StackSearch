from os import path


class HParams:
    filePath = path.join("data", "partial", "Posts.xml")
    BATCH_SIZE = 4
    OUTPUT_DIM = 64
    CKPT_MAX_TO_KEEP = 2
    MAX_SENTENCE_DIM = 30




    def getCeckpointPath(self):
        return str(self.OUTPUT_DIM)




