from index_builder.faiss_hnsw.module import FaissHNSW
import faiss
import numpy as np
import torch
import tqdm
import DB_build.dataset as dataset
from datasets import load_dataset
import os
from sentence_transformers import SentenceTransformer

# model_set = ["multi-qa-mpnet-base-cos-v1", "multi-qa-distilbert-cos-v1", "multi-qa-MiniLM-L6-cos-v1"]
model_set = ["multi-qa-MiniLM-L6-cos-v1"]
# dataset_list = ["msmarco", "nq", "hotpotqa", "arguana", "webis-touche2020", "dbpedia-entity","fever"]
dataset_list = ["nq"]


print("starting...")
for dataset_name in dataset_list:
    for model_config in model_set:
        print(f"inserting {dataset_name} to {model_config} DB")
        index = FaissHNSW("angular", {"M": 64, "efConstruction": 1024})

        corpus = dataset.DATASET_BUILDERS["BeIR/"+dataset_name +"/corpus"].build(load_dataset("BeIR/" + dataset_name, 'corpus'))
        vectors = np.load("embeddings" + "/"+ dataset_name + "/"+ model_config + "_all.npy")
        print("load compelet")

        print("corpus and vectors are ready to insert")

        index.fit(vectors, corpus["doc-id"])
        index.set_query_arguments(1024)
        print("index created")

        # path = "faiss_DB/" + dataset_name
        # if not os.path.exists(path):
        #     os.makedirs(path)
        index.save_index("/home/work/mintaek2/faiss_DB/"+dataset_name +"/" + model_config + "_migrate_tuned.bin")
        print("save done")