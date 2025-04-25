import pickle
import random
import torch
from mapping_function import Fix_MLP
from pymilvus import model
from index_builder.faiss_hnsw.module import FaissHNSW
from ../HVMap import module

model_from = "multi-qa-mpnet-base-cos-v1"
model_to = "multi-qa-distilbert-cos-v1"
dataset_name = "nq"
k = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

index_from = FaissHNSW.load_index("angular", {"M": 32, "efConstruction": 128},
                                  f"faiss_DB/{dataset_name}/{model_from}_firsthalf.bin")
index_to = FaissHNSW.load_index("angular", {"M": 32, "efConstruction": 128},
                                f"faiss_DB/{dataset_name}/{model_to}_secondalf.bin")

embedding_model_from = model.dense.SentenceTransformerEmbeddingFunction(model_name=model_from, device=device)
embedding_model_to = model.dense.SentenceTransformerEmbeddingFunction(model_name=model_to, device=device)

converter = Fix_MLP(768, 768)
converter.load_state_dict(torch.load(f"test_models/{model_from}--TO--{model_to}/trainset_model.pth", map_location=device))

converter2 = Fix_MLP(768, 768)
converter2.load_state_dict(torch.load(f"test_models/{model_to}--TO--{model_from}/trainset_model.pth", map_location=device))

with open(f"{dataset_name}.pickle", 'rb') as f:
    test_set = pickle.load(f)

if len(test_set) >= 1000:
    q_text = random.sample(list(test_set.keys()), 1000)
else:
    keys = list(test_set.keys())
    q_text = keys + random.choices(keys, k=1000 - len(keys))

hvmap = HVMap(
    model_from=model_from,
    model_to=model_to,
    dataset_name=dataset_name,
    embedding_model_from=embedding_model_from,
    embedding_model_to=embedding_model_to,
    index_from=index_from,
    index_to=index_to,
    converter=converter,
    converter2=converter2,
    q_text=q_text,
    k=k,
    device=device
)

results = hvmap.query(converting_parm=0.75)

print("Top-k result for first query:")
print(results[0])
