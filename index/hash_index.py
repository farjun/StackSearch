from scipy.spatial import distance
import pickle
import os
import heapq


class Index:

    def __init__(self, index_dir_path, threshold=1000, distance_func=distance.hamming, key_extractor=lambda _: _[0]):
        self.key_extractor = key_extractor  # can be adapted to the key structure
        self.main_index_path = os.path.join(index_dir_path, "main_index")
        self.blocks_index_path = os.path.join(index_dir_path, "blocks_index")
        self.sorted_main_index_path = os.path.join(index_dir_path, "sorted_index")
        self.merged_blocks_index_path = os.path.join(index_dir_path, "merged_blocks_index")
        if not os.path.exists(self.sorted_main_index_path):
            with open(self.sorted_main_index_path, 'wb') as f:
                f.write(b"")  # create file
        if not os.path.exists(self.main_index_path):
            with open(self.main_index_path, 'wb') as f:
                f.write(b"")  # create file
        if not os.path.exists(self.blocks_index_path):
            with open(self.blocks_index_path, 'wb') as f:
                f.write(b"")  # create file
            self.blocks_start_index = [0]
        if not os.path.exists(self.merged_blocks_index_path):
            with open(self.merged_blocks_index_path, 'wb') as f:
                f.write(b"")  # create file
            self.merged_blocks_start_index = []
        else:
            with open(self.merged_blocks_index_path, "rb") as f:
                self.merged_blocks_start_index = pickle.load(f)

        self.distance_func = distance_func
        self.block = []
        self.threshold = threshold
        self.in_chunk_index = [0]
        self.CHUNK_SIZE = 100  # arbitrary for now

    def clean_index(self):
        os.remove(self.main_index_path)
        os.remove(self.blocks_index_path)

    def insert(self, key, val):
        self.block.append((key, val))
        if len(self.block) > self.threshold:
            self._dump_block()

    def brute_force_search(self, key, dist_limit=0, result_size_limit=10):
        res = []
        for b_num in range(len(self.merged_blocks_start_index)):
            self._load_block(b_num)
            for item in self.block:
                if self.distance_func(item[0], key) <= dist_limit:
                    res.append(item)
                    if len(res) == result_size_limit:
                        return res
        return res

    def search(self, key, result_size_limit=10):
        arr = self.merged_blocks_start_index
        left = 0
        right = len(arr) - 1
        while left <= right:
            mid = left + right / 2
            mid_ind = int(mid)
            if key < arr[mid_ind][1]:
                right = mid - 1
            elif key > arr[mid_ind][1]:
                left = mid + 1
            else:
                break
    #  TODO make it smarter in finding surrounding values
        self._load_block(mid_ind)
        return heapq.nsmallest(result_size_limit, self.block, key=lambda k: self.distance_func(self.key_extractor(k), key))

    def _dump_block(self):
        with open(self.main_index_path, 'ab') as f:
            self.block.sort()
            serialized = pickle.dumps(self.block, protocol=pickle.HIGHEST_PROTOCOL)
            f.write(serialized)
            self.blocks_start_index.append(f.tell())
            self.in_chunk_index.append(0)
        self.block = []

    def _load_block(self, block_num, index_path=None, blocks_start_index=None):
        index_path = self.sorted_main_index_path
        blocks_start_index = self.merged_blocks_start_index
        ind = block_num + 1
        self.block = []
        with open(index_path, 'rb') as f:
            f.seek(blocks_start_index[block_num][0])
            if ind < len(blocks_start_index):
                serialized_block = f.read(blocks_start_index[ind][0])
            else:
                serialized_block = f.read()
            if len(serialized_block) > 0:
                self.block = pickle.loads(serialized_block)

    def _get_chunks_file_objects(self):
        file_objects = []
        f = open(self.main_index_path, 'rb')
        f.seek(0)
        file_objects.append(f)
        for i, block_start in enumerate(self.blocks_start_index):
            f = open(self.main_index_path, 'rb')
            f.seek(block_start)
            file_objects.append(f)
        return file_objects

    def _get_chunks(self, file_objects):
        chunks = []
        for i, block_start in enumerate(self.blocks_start_index):
            file_object = file_objects[i]
            full_chunk = file_object.read(block_start) if i < len(self.blocks_start_index) - 1 else file_object.read()
            if len(full_chunk) > 0:
                full_chunk = pickle.loads(full_chunk)
                if len(full_chunk) > self.in_chunk_index[i]:
                    chunks.append(full_chunk[self.in_chunk_index[i]:self.in_chunk_index[i] + self.CHUNK_SIZE])
                    del full_chunk
                    self.in_chunk_index[i] += self.CHUNK_SIZE
        return chunks

    def sort(self):
        chunks_file_objects = self._get_chunks_file_objects()
        sorted_main_index_file = open(self.sorted_main_index_path, 'ab')
        merged_blocks_index_file = open(self.merged_blocks_index_path, 'wb')
        last_block_end = 0
        try:
            while True:
                chunks = self._get_chunks(chunks_file_objects)
                if len(chunks) == 0:
                    pickle.dump(self.merged_blocks_start_index, merged_blocks_index_file)
                    return
                merged_block = list(heapq.merge(*chunks, key=self.key_extractor))
                serialized = pickle.dumps(merged_block, protocol=pickle.HIGHEST_PROTOCOL)
                sorted_main_index_file.write(serialized)
                self.merged_blocks_start_index.append((last_block_end, self.key_extractor(merged_block[0])))
                last_block_end = sorted_main_index_file.tell()
        finally:
            for f in chunks_file_objects:
                f.close()
            self.clean_index()
            sorted_main_index_file.close()
            merged_blocks_index_file.close()

    def print(self):
        for b_num in range(len(self.merged_blocks_start_index)):
            self._load_block(b_num)
            for item in self.block:
                print(item)
