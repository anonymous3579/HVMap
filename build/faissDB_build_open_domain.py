from index_builder.faiss_hnsw.module import FaissHNSW
import faiss
import numpy as np
import torch
import tqdm
import DB_build.dataset as dataset
from datasets import load_dataset
import os
from sentence_transformers import SentenceTransformer

model_set = ["multi-qa-mpnet-base-cos-v1", "multi-qa-distilbert-cos-v1", "multi-qa-MiniLM-L6-cos-v1"]
dataset_list = [ "nq", "hotpotqa", "arguana", "webis-touche2020", "dbpedia-entity","fever", "msmarco"]


print("starting...")
for dataset_name in dataset_list:
    for model_config in model_set:
        print(f"inserting {dataset_name} to {model_config} DB")
        index_even = FaissHNSW("angular", {"M": 32, "efConstruction": 128})
        index_odd = FaissHNSW("angular", {"M": 32, "efConstruction": 128})

        corpus = dataset.DATASET_BUILDERS["BeIR/"+dataset_name +"/corpus"].build(load_dataset("BeIR/" + dataset_name, 'corpus'))
        vectors = np.load("embeddings" + "/"+ dataset_name + "/"+ model_config + "_all.npy")
        print("load compelet")
        doc_id_even = corpus[:len(corpus)//2]["doc-id"]
        doc_id_odd = corpus[len(corpus)//2:]["doc-id"]
        vectors_even = vectors[:len(corpus)//2]
        vectors_odd = vectors[len(corpus)//2:]  
        print("corpus and vectors are ready to insert")

        index_even.fit(vectors_even, doc_id_even)
        index_even.set_query_arguments(128)
        index_odd.fit(vectors_odd, doc_id_odd)
        index_odd.set_query_arguments(128)
        print("index created")

        path = "faiss_DB/" + dataset_name
        if not os.path.exists(path):
            os.makedirs(path)
        index_even.save_index("faiss_DB/"+dataset_name +"/" + model_config + "_firsthalf.bin")

        index_odd.save_index("faiss_DB/"+dataset_name +"/" + model_config + "_secondalf.bin")
        print("save done")