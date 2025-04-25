import os
import numpy as np


# dataset_list = ["msmarco"]
dataset_list = ["msmarco","nq", "hotpotqa", "arguana","dbpedia-entity","fever"]
model_set = ["multi-qa-mpnet-base-cos-v1", "multi-qa-distilbert-cos-v1", "multi-qa-MiniLM-L6-cos-v1"]
model_set = ["fin_model", "sci_model", "bio_model"]
sample_size = 500000

for model_name in model_set:
    print(f"{model_name} starts")
    combined_samples = []
    for dataset_name in dataset_list:
        print(f"{dataset_name} on {model_name} space")
        path = dataset_name + '/' + model_name + "_all.npy"
        data = np.load(path)
        np.random.seed(42)
        if len(data) > sample_size:
                sampled_data = data[np.random.choice(len(data), size=sample_size, replace=False)]
        else:
            sampled_data = data
        print(f"size of data: {len(sampled_data)}")
        combined_samples.append(sampled_data)

    result = np.concatenate(combined_samples, axis=0)
    print(f"Combined array shape: {result.shape}")
    output_path = "/embeddings/trainset/" + model_name

    output_path = output_path + "_combine.npy"
    np.save(output_path, result)
    print(f"Saved combined array to {output_path}")

