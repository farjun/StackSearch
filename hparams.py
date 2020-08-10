from os import path


class HParams:
    filePath = path.join("data", "partial", "Posts.xml")
    BATCH_SIZE = 4
    OUTPUT_DIM = 32




    def getCeckpointPath(self):
        return str(self.OUTPUT_DIM)



