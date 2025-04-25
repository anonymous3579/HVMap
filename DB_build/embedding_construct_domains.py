from pymilvus import MilvusClient
from pymilvus import model
import torch
import numpy as np  # numpy 라이브러리 임포트
import tqdm
import dataset
from datasets import load_dataset
import os
model_set = ["multi-qa-mpnet-base-cos-v1", "multi-qa-distilbert-cos-v1", "multi-qa-MiniLM-L6-cos-v1"]
dataset_list = ["msmarco", "nq", "hotpotqa", "fiqa", "nfcorpus", "arguana", "webis-touche2020", "dbpedia-entity", "scidocs", "fever","scifact", "trec-covid"]
####################
model_config = model_set[2]
data_size = "all"
similarity_metirc = "COSINE"
#####################
# hotpot은 doc id 530496 일때 200000만개


from transformers import AutoTokenizer, AutoModel

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



embedding_model_bio = model.dense.SentenceTransformerEmbeddingFunction(
        model_name="juanpablomesa/bge-base-bioasq-matryoshka",  # 모델 이름 지정
        device=device  # 디바이스 지정 ('cpu' 또는 'cuda:0')
    )

embedding_model_fin = CustomSentenceTransformer("rbhatia46/financial-rag-matryoshka")

# hetero embedding model load
embedding_model_sci = model.dense.SentenceTransformerEmbeddingFunction(
    model_name="pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT", 
    device=device  
)



# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_config in model_set:
    corpus = dataset.DATASET_BUILDERS["BeIR/msmarco/queries"].build(load_dataset("BeIR/msmarco", 'queries'))
    print("size of corpus: ",len(corpus))
    try:
        docs = corpus.select(range(data_size)) ## 이거 뭔가 이상하다 
    except:
        docs = corpus

    print(f"embedding with {model_config}")
    embedding_model = model.dense.SentenceTransformerEmbeddingFunction(
        model_name=model_config,  # 모델 이름 지정
        device=device  # 디바이스 지정 ('cpu' 또는 'cuda:0')
    )

    npy_dic_path = '../embeddings/msmarco_queries'
    if not os.path.exists(npy_dic_path):
        os.makedirs(npy_dic_path)
    try :
        vectors = np.load(npy_dic_path + "/" + model_config + "_all.npy")
    except :
        print("constructing embeddings")
        # 배치 설정
        batch_size = 3000  # 원하는 배치 크기 지정
        num_batches = len(docs) // batch_size + (1 if len(docs) % batch_size != 0 else 0)

        # 벡터 임베딩 생성 및 진행 상태 표시
        all_vectors = []
        for i in tqdm.tqdm(range(num_batches), desc="Encoding batches"):
            batch_docs = docs[i * batch_size: (i + 1) * batch_size]['text']
            batch_vectors = embedding_model.encode_documents(batch_docs)
            all_vectors.append(batch_vectors)

        # 배치로 생성된 벡터 합치기
        vectors = np.vstack(all_vectors)

    print("finish constructing vectors")
    # 벡터를 npy 파일로 저장
    npy_file_path = npy_dic_path + "/" + model_config + '_all.npy'  # 저장할 파일 경로와 이름
    np.save(npy_file_path, vectors)  # vectors를 npy 파일로 저장

    print(f'vectors of {model_config} space are saved')