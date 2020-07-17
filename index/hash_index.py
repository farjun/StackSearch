from scipy.spatial import distance
from math import ceil
import pickle
import os
import heapq
from logging import error


def erase_on_raised_exception(f):
    def new_f(self, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except Exception as e:
            error('Raised from {}:'.format(f))
            error(e)
            erase_flag = getattr(self, 'to_erase_on_raised_exception', False)
            if erase_flag:
                self.erase_all()
            raise e

    return new_f


class Index:

    def __init__(self, index_dir_path, disk_chunk_size=1000, distance_func=distance.hamming,
                 key_extractor=lambda _: _[0], to_erase_on_raised_exception=True):

        self.to_erase_on_raised_exception = to_erase_on_raised_exception
        self.key_extractor = key_extractor  # can be adapted to the key structure
        self.main_index_path = os.path.join(index_dir_path, "main_index")
        self.blocks_index_path = os.path.join(index_dir_path, "blocks_index")
        self.sorted_main_index_path = os.path.join(index_dir_path, "sorted_index")
        self.merged_blocks_index_path = os.path.join(index_dir_path, "merged_blocks_index")
        if not os.path.exists(index_dir_path):
            os.makedirs(index_dir_path)

        if not os.path.exists(self.sorted_main_index_path):
            with open(self.sorted_main_index_path, 'wb') as f:
                f.write(b"")  # create file
        if not os.path.exists(self.main_index_path):
            with open(self.main_index_path, 'wb') as f:
                f.write(b"")  # create file
        if not os.path.exists(self.blocks_index_path):
            with open(self.blocks_index_path, 'wb') as f:
                f.write(b"")  # create file
            self.blocks_start_index = []
        if not os.path.exists(self.merged_blocks_index_path):
            with open(self.merged_blocks_index_path, 'wb') as f:
                f.write(b"")  # create file
            self.merged_blocks_start_index = []
        else:
            with open(self.merged_blocks_index_path, "rb") as f:
                self.merged_blocks_start_index = pickle.load(f)

        self.distance_func = distance_func
        self.block = []
        self.disk_chunk_size = disk_chunk_size
        self.in_disk_chunk_index = []

    def erase_all(self):
        self.clean_index()
        if os.path.exists(self.merged_blocks_index_path):
            os.remove(self.merged_blocks_index_path)
        if os.path.exists(self.sorted_main_index_path):
            os.remove(self.sorted_main_index_path)

    def clean_index(self):
        if os.path.exists(self.blocks_index_path):
            os.remove(self.blocks_index_path)
        if os.path.exists(self.main_index_path):
            os.remove(self.main_index_path)

    @erase_on_raised_exception
    def insert(self, key, val):
        self.block.append((key, val))
        if len(self.block) > self.disk_chunk_size:
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
        mid_ind = 0
        while left < right:
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
        return heapq.nsmallest(result_size_limit, self.block,
                               key=lambda k: self.distance_func(self.key_extractor(k), key))

    @erase_on_raised_exception
    def _dump_block(self):
        if len(self.block) == 0:
            return
        if not hasattr(self, "_last_block_end"):
            self._last_block_end = 0
        with open(self.main_index_path, 'ab') as f:
            self.block.sort()
            # print('DUMPED------{}'.format(self.block))
            serialized = pickle.dumps(self.block, protocol=pickle.HIGHEST_PROTOCOL)
            f.write(serialized)
            self.blocks_start_index.append(self._last_block_end)
            self._last_block_end = f.tell()
            self.in_disk_chunk_index.append(0)
        self.block = []

    @erase_on_raised_exception
    def _load_block(self, block_num):
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

    @erase_on_raised_exception
    def _get_chunks_file_objects(self):
        file_objects = []
        for block_start in self.blocks_start_index:
            f = open(self.main_index_path, 'rb')
            f.seek(block_start)
            file_objects.append(f)
        return file_objects

    @erase_on_raised_exception
    def _get_chunks(self, file_objects, merger_chunk_size):
        chunks = []
        for i in range(len(self.blocks_start_index)):
            file_object = file_objects[i]
            block_length = self.blocks_start_index[i+1] if i < len(self.blocks_start_index) - 1 else None
            full_chunk = file_object.read(block_length) if block_length else file_object.read()
            if len(full_chunk) > 0:
                full_chunk = pickle.loads(full_chunk)
                if len(full_chunk) > self.in_disk_chunk_index[i]:
                    chunks.append(full_chunk[self.in_disk_chunk_index[i]:self.in_disk_chunk_index[i]+merger_chunk_size])
                    del full_chunk
                    self.in_disk_chunk_index[i] += merger_chunk_size
        return chunks

    @erase_on_raised_exception
    def sort(self):
        self._dump_block()  # dumping any leftovers
        sorted_main_index_file = open(self.sorted_main_index_path, 'ab')
        merged_blocks_index_file = open(self.merged_blocks_index_path, 'wb')
        last_block_end = 0
        merger_chunk_size = ceil(self.disk_chunk_size/max(1, len(self.blocks_start_index)))
        try:
            while True:
                chunks_file_objects = self._get_chunks_file_objects()
                if chunks_file_objects is None:
                    break
                chunks = self._get_chunks(chunks_file_objects, merger_chunk_size)
                # print('CHUNKS TO MERGE----{}'.format(chunks))
                if len(chunks) == 0:
                    pickle.dump(self.merged_blocks_start_index, merged_blocks_index_file)
                    return
                merged_block = list(heapq.merge(*chunks, key=self.key_extractor))
                print('MERGED--------{}'.format(merged_block))
                serialized = pickle.dumps(merged_block, protocol=pickle.HIGHEST_PROTOCOL)
                sorted_main_index_file.write(serialized)
                self.merged_blocks_start_index.append((last_block_end, self.key_extractor(merged_block[0])))
                last_block_end = sorted_main_index_file.tell()
        except Exception as e:
            print(e)
        finally:
            if chunks_file_objects is not None:
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
