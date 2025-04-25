from pymilvus import MilvusClient
from pymilvus import model
import torch
import numpy as np  # numpy 라이브러리 임포트
import tqdm
import dataset
from datasets import load_dataset
import os
model_set = ["multi-qa-mpnet-base-cos-v1", "multi-qa-distilbert-cos-v1", "multi-qa-MiniLM-L6-cos-v1"]
dataset_list = ["msmarco", "nq", "hotpotqa","arguana","dbpedia-entity","fever"]
####################
model_config = model_set[3]
data_size = "all"
similarity_metirc = "COSINE"
#####################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for dataset_name in dataset_list:
    corpus = dataset.DATASET_BUILDERS["BeIR/"+dataset_name +"/corpus"].build(load_dataset("BeIR/" + dataset_name, 'corpus'))
    print("size of corpus: ",len(corpus))
    try:
        docs = corpus.select(range(data_size))
    except:
        docs = corpus
        
    print(f"embedding {dataset_name} with {model_config}")
    embedding_model = model.dense.SentenceTransformerEmbeddingFunction(
        model_name=model_config, 
        device=device 
    )

    npy_dic_path = '../embeddings/'+ dataset_name
    if not os.path.exists(npy_dic_path):
        os.makedirs(npy_dic_path)
    try :
        vectors = np.load(npy_dic_path + "/" + model_config + "_all.npy")
    except :
        print("constructing embeddings")
        batch_size = 3000
        num_batches = len(docs) // batch_size + (1 if len(docs) % batch_size != 0 else 0)

        
        all_vectors = []
        for i in tqdm.tqdm(range(num_batches), desc="Encoding batches"):
            batch_docs = docs[i * batch_size: (i + 1) * batch_size]['text']
            batch_vectors = embedding_model.encode_documents(batch_docs)
            all_vectors.append(batch_vectors)

        vectors = np.vstack(all_vectors)

    print("finish constructing vectors")
    npy_dic_path = '../embeddings/'+ dataset_name
    npy_file_path = npy_dic_path + "/" + model_config + '_all.npy'
    np.save(npy_file_path, vectors)

    print(f'vectors of {dataset_name} on {model_config} space are saved')