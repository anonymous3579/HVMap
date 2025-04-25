from pymilvus import MilvusClient
from pymilvus import model
import torch
import numpy as np  # numpy 라이브러리 임포트
import tqdm
import dataset
from datasets import load_dataset
import os
from sentence_transformers import SentenceTransformer
model_set = ['financial-rag-matryoshka']

####################
# model_config = model_set[0]
# data_size = 200000
# dataset_name = "msmarco"
# dataset_name = "trec-covid"
# dataset_name = "scifact"
dataset_name = "scidocs"
similarity_metirc = "COSINE"
#####################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################################################################################################
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
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



##########################################################################################################

corpus = dataset.DATASET_BUILDERS["BeIR/"+dataset_name +"/corpus"].build(load_dataset("BeIR/" + dataset_name, 'corpus'))
print("size of corpus: ",len(corpus))
try:
    docs = corpus.select(range(data_size)) ## 이거 뭔가 이상하다 
except:
    docs = corpus

# print(docs[-1])

for model_config in model_set:
    # # SentenceTransformer 모델 로드
    # embedding_model = model.dense.SentenceTransformerEmbeddingFunction(
    #     model_name="juanpablomesa/bge-base-bioasq-matryoshka",  # 모델 이름 지정
    #     device=device  # 디바이스 지정 ('cpu' 또는 'cuda:0')
    # )
    embedding_model = CustomSentenceTransformer("rbhatia46/financial-rag-matryoshka")
    dimension = len(embedding_model.encode_queries(["test sentence"])[0])
    

    

    npy_dic_path = '../embeddings/'+ dataset_name
    if not os.path.exists(npy_dic_path):
        os.makedirs(npy_dic_path)

    try :
        vectors = np.load(npy_dic_path + "/" + model_config + ".npy")
    except :
        print("constructing embeddings")
        # 배치 설정
        batch_size = 1000  # 원하는 배치 크기 지정
        num_batches = len(docs) // batch_size + (1 if len(docs) % batch_size != 0 else 0)

        # 벡터 임베딩 생성 및 진행 상태 표시
        all_vectors = []
        for i in tqdm.tqdm(range(num_batches), desc="Encoding batches"):
            batch_docs = docs[i * batch_size: (i + 1) * batch_size]['text']
            batch_vectors = embedding_model.encode_documents(batch_docs)
            # batch_vectors = embedding_model.encode(batch_docs)
            all_vectors.append(batch_vectors)

        # 배치로 생성된 벡터 합치기
        vectors = np.vstack(all_vectors)

    # 벡터를 npy 파일로 저장
    npy_dic_path = '../embeddings/'+ dataset_name
    npy_file_path = npy_dic_path + "/" + model_config + '.npy'  # 저장할 파일 경로와 이름
    np.save(npy_file_path, vectors)  # vectors를 npy 파일로 저장

    print('vectors are ready')

    directory = "../DB/" + dataset_name
    if not os.path.exists(directory):
        os.makedirs(directory)


    DB_path = directory + '/' + model_config + ".db"
    client = MilvusClient(DB_path)
    if client.has_collection(collection_name="collection"):
        client.drop_collection(collection_name="collection")
    client.create_collection(
        collection_name="collection",
        dimension=dimension,
        metric_type=similarity_metirc
    )

    data = [
        {"id": i, "vector": vectors[i], "text": cor['text'], "doc_id": cor['doc-id']}
        for i, cor in tqdm.tqdm(enumerate(docs))
    ]

    print('data is ready')

    BATCH_SIZE = 1000  # 배치 크기를 설정

    for i in tqdm.tqdm(range(0, len(vectors), BATCH_SIZE)):
        batch_data = data[i:i + BATCH_SIZE]
        client.insert(collection_name="collection", data=batch_data)
        # print(f"Inserted batch {i} to {i + BATCH_SIZE}")


    print("Finished saving DB")