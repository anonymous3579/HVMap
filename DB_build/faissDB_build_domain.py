from index_builder.faiss_hnsw.module import FaissHNSW
import faiss
import numpy as np
import torch
import tqdm
import DB_build.dataset as dataset
from datasets import load_dataset
import os
from sentence_transformers import SentenceTransformer

model_set = ["sci_model", "fin_model", "bio_model"]
# model_set = ["multi-qa-distilbert-cos-v1"]
# dataset_list = ["arguana", "webis-touche2020", "dbpedia-entity","fever", "msmarco",  "nq", "hotpotqa"]
dataset_list = ["bioasq"]
dataset_list = [ "fiqa", "scifact", "scidocs"]
# dataset_list = ["dbpedia-entity"]

# dataset_list = ["nq", "hotpotqa", "arguana", "webis-touche2020", "dbpedia-entity","fever"]

print("starting...")
for dataset_name in dataset_list:
    for model_config in model_set:
        print(f"inserting {dataset_name} to {model_config} DB")
        index_even = FaissHNSW("angular", {"M": 32, "efConstruction": 128})
        index_odd = FaissHNSW("angular", {"M": 32, "efConstruction": 128})

        corpus = dataset.DATASET_BUILDERS["BeIR/"+dataset_name +"/corpus"].build(load_dataset("BeIR/" + dataset_name, 'corpus'))
        vectors = np.load("/home/work/mintaek2/embeddings" + "/"+ dataset_name + "/"+ model_config + "_all.npy")
        print("load compelet")
        doc_id_even = corpus[:len(corpus)//4]["doc-id"]
        doc_id_odd = corpus[len(corpus)//4:]["doc-id"]
        vectors_even = vectors[:len(corpus)//4]
        vectors_odd = vectors[len(corpus)//4:]  
        print("corpus and vectors are ready to insert")

        path = "/home/work/mintaek2/faiss_DB/" + dataset_name
        if not os.path.exists(path):
            os.makedirs(path)


        index_odd.fit(vectors_odd, doc_id_odd)
        index_odd.set_query_arguments(128)
        index_odd.save_index(path + "/" + model_config + "_therest.bin")
        print("index created")

        index_even.fit(vectors_even, doc_id_even)
        index_even.set_query_arguments(128)
        print("index created")
        index_even.save_index(path +"/" + model_config + "_firstquarter.bin")
        
        


        # if not os.path.exists(path):
        #     os.makedirs(path)
        

        index_odd.save_index(path + "/" + model_config + "_therest.bin")
        print("save done")