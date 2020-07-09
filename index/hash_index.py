from scipy.spatial import distance
import pickle
import os


class Index:

    def __init__(self, index_dir_path, threshold=1000, distance_func=distance.hamming):
        self.main_index = os.path.join(index_dir_path, "main_index")
        self.blocks_index = os.path.join(index_dir_path, "blocks_index")
        if not os.path.exists(self.main_index):
            with open(self.main_index, 'wb') as f:
                f.write(b"")  # create file
        if not os.path.exists(self.blocks_index):
            with open(self.blocks_index, 'wb') as f:
                f.write(b"")  # create file
            self.blocks_start_index = [0]
        else:
            with open(self.blocks_index, "rb") as f:
                self.blocks_start_index = pickle.load(f)

        self.distance_func = distance_func
        self.block = []
        self.threshold = threshold

    # def __del__(self):
    #     with open(self.blocks_index, "wb") as f:
    #         pickle.dump(self.blocks_start_index, f)

    def insert(self, key, val):
        self.block.append((key, val))
        if len(self.block) > self.threshold:
            self._dump_block()

    def _cluster_blocks(self):
        pass

    def search(self, key, dist_limit=0):
        res = []
        for b_num in range(len(self.blocks_start_index)):
            self._load_block(b_num)
            for item in self.block:
                if self.distance_func(item[0], key) <= dist_limit:
                    res.append(item)
        return res

    def _dump_block(self):
        with open(self.main_index, 'ab') as f:
            serialized = pickle.dumps(self.block, protocol=pickle.HIGHEST_PROTOCOL)
            f.write(serialized)
            self.blocks_start_index.append(f.tell())
        self.block = []

    def _load_block(self, block_num):
        ind = block_num + 1
        self.block = []
        with open(self.main_index, 'rb') as f:
            f.seek(self.blocks_start_index[block_num])
            if ind < len(self.blocks_start_index):
                serialized_block = f.read(self.blocks_start_index[ind])
            else:
                serialized_block = f.read()
            if len(serialized_block) > 0:
                self.block = pickle.loads(serialized_block)

