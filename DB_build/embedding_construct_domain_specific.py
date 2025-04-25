from index_builder.faiss_hnsw.module import FaissHNSW
import faiss
import pickle
import numpy as np
import torch
import tqdm as tqdm
from mapping_function import *
from pymilvus import model
import dataset as dataset
from datasets import load_dataset, concatenate_datasets, disable_progress_bar
import torch.nn.functional as F
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import random
dataset_list = ["msmarco", "nq", "hotpotqa","arguana","dbpedia-entity","fever"]
data_size = "all"
similarity_metirc = "COSINE"
#####################

class CustomSentenceTransformer:
    def __init__(self, model_name):
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def encode_queries(self, sentences, batch_size=8):
        self.model.eval()
        embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling over the token embeddings
            # Assumes `last_hidden_state` is the token embeddings from the model
            token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
            attention_mask = inputs['attention_mask']
            
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
            
            embeddings.append(sentence_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    def encode_documents(self, sentences, batch_size=8):
        self.model.eval()
        embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling over the token embeddings
            # Assumes `last_hidden_state` is the token embeddings from the model
            token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
            attention_mask = inputs['attention_mask']
            
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
            
            embeddings.append(sentence_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_model_fin = CustomSentenceTransformer("rbhatia46/financial-rag-matryoshka")

embedding_model_bio = model.dense.SentenceTransformerEmbeddingFunction(
        model_name="juanpablomesa/bge-base-bioasq-matryoshka",
        device=device
    )

embedding_model_sci = model.dense.SentenceTransformerEmbeddingFunction(
    model_name="pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT", 
    device=device  
)


model_config = "fin_model"
# model_config = "bio_model"
# model_config = "sci_model"


if model_config == "fin_model":
    embedding_model = embedding_model_fin
elif model_config == "bio_model":
    embedding_model = embedding_model_bio
elif model_config == "sci_model":
    embedding_model = embedding_model_sci

for dataset_name in dataset_list:
    corpus = dataset.DATASET_BUILDERS["BeIR/"+dataset_name +"/corpus"].build(load_dataset("BeIR/" + dataset_name, 'corpus'))
    print("size of corpus: ",len(corpus))
    try:
        docs = corpus.select(range(data_size)) 
    except:
        docs = corpus
        
    print(f"embedding {dataset_name} with {model_config}")
    

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
        npy_file_path = npy_dic_path + "/" + model_config + '_all.npy'
        np.save(npy_file_path, vectors)

    print(f'vectors of {dataset_name} on {model_config} space are saved')