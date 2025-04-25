import faiss
import numpy as np

import faiss
import numpy as np
import pickle


class FaissHNSW:

    class GeneralIdMapper:
        def __init__(self):
            self.__key_to_id = dict()
            self.__id_to_key = dict()
            self.__count = 0

        def get_ids(self, keys: list):
            ids = np.empty(len(keys), dtype=int)
            for i, key in enumerate(keys):
                ids[i] = self.get_id(key)
            return ids

        def get_id(self, key):
            if key not in self.__key_to_id:
                self.__key_to_id[key] = self.__count
                self.__id_to_key[self.__count] = key
                self.__count += 1
            return self.__key_to_id[key]

        def get_key(self, id):
            return self.__id_to_key.get(id, None)

        def to_dict(self):
            return {
                "key_to_id": self.__key_to_id,
                "id_to_key": self.__id_to_key,
                "count": self.__count,
            }

        def from_dict(self, state):
            self.__key_to_id = state["key_to_id"]
            self.__id_to_key = state["id_to_key"]
            self.__count = state["count"]

    def __init__(self, metric, method_param):
        self.__generalIdMapper = self.GeneralIdMapper()
        self._metric = metric
        self.method_param = method_param
        self.index = None

    def query(self, v, k):  
        if self._metric == "angular":
            v = v / np.linalg.norm(v)
        D, I = self.index.search(np.expand_dims(v, axis=0).astype(np.float32), k)
        V = np.array([self.index.reconstruct(int(idx)) for idx in I[0]])
        doc_keys = np.array([self.__generalIdMapper.get_key(id) for id in I[0]])
        return [doc_keys, V]
    
    def batch_query(self, X, k):
        if self._metric == "angular":
            X /= np.linalg.norm(X)
        D, L = self.index.search(X.astype(np.float32), k)
        res = []
        for i in range(len(D)):
            r = []
            for l, d in zip(L[i], D[i]):
                if l != -1:
                    r.append(l)
            res.append(r)
        return res

    def fit(self, X, data_labels):  
        if self.index is None:
            self.index = faiss.IndexHNSWFlat(len(X[0]), self.method_param["M"])
            self.index.hnsw.efConstruction = self.method_param["efConstruction"]
            self.index.verbose = True
            self.index = faiss.IndexIDMap2(self.index)

        if self._metric == "angular":
            X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        ids = self.__generalIdMapper.get_ids(data_labels)
        self.index.add_with_ids(X, ids)
        faiss.omp_set_num_threads(1)

    def set_query_arguments(self, ef):
        faiss.cvar.hnsw_stats.reset()
        self.index.index.efSearch = ef  #

    def save_index(self, path: str):
        # 인덱스와 GeneralIdMapper 상태 저장
        faiss.write_index(self.index, path + ".index")
        with open(path + ".mapper", "wb") as f:
            pickle.dump(self.__generalIdMapper.to_dict(), f)

    def get_id(self, key):
        return self.__generalIdMapper.get_id(key)

    def get_vector_by_key(self, key):
        return self.index.reconstruct(self.get_id(key))

    def get_vector_by_id(self, id):
        return self.index.reconstruct(id)

    @classmethod
    def load_index(cls, metric, method_param, path: str):
        # 인덱스와 GeneralIdMapper 상태 로드
        index = faiss.read_index(path + ".index")
        with open(path + ".mapper", "rb") as f:
            mapper_state = pickle.load(f)

        instance = cls(metric, method_param)
        instance.index = index
        instance.__generalIdMapper.from_dict(mapper_state)
        return instance

    def __str__(self):
        efSearch = self.index.index.efSearch  # 수정된 부분
        return f"faiss ({self.method_param}, ef: {efSearch})"

    def freeIndex(self):
        del self.__generalIdMapper
        del self.index


if __name__ == "__main__":
    index = FaissHNSW("angular", {"M": 32, "efConstruction": 128})
    ids = np.arange(10000)
    np.random.seed(13)
    X = np.random.rand(10000, 128)

    print(X.shape)
    index.fit(X, ids)
    index.set_query_arguments(128)

    # END OF SETUP
    k = 10
    query = np.random.rand(128)
    print(query.shape)
    ret = index.query(query, k)
    print(ret)
    key = ret[0][0]
    print(f"key: {key}")
    print(f"id: {index.get_id(key)}")
    print(f"vector : {index.get_vector_by_key(key)}")
    print(f"vector : {index.get_vector_by_id(index.get_id(key))}")

    # 인덱스 저장
    index.save_index("faiss_hnsw_index.bin")

    # 인덱스 로드
    loaded_index = FaissHNSW.load_index("angular", {"M": 32, "efConstruction": 128}, "faiss_hnsw_index.bin")
    loaded_index.set_query_arguments(128)