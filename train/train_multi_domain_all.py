import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from mapping_function import *
from tqdm import tqdm
import torch.nn.functional as F
import os
from pymilvus import model

##################################
######## Hyperparameters #########
##################################
batch_size = 1792
learning_rate = 0.001
num_epochs = 30
model_set = ["bio_model", "fin_model", "sci_model"]
dataset_name = "trainset"
###################################
###################################

def fuzzy_loss(x, y):
    # f_x와 y의 코사인 유사도 계산
    sim_ij = F.cosine_similarity(y.unsqueeze(1), y.unsqueeze(0), dim=-1)
    f_x_y = F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=-1)
    loss = torch.sum((sim_ij - f_x_y) ** 2) / len(x) / len(y)
    return loss

# Custom Dataset class
class VectorDataset(Dataset):
    def __init__(self, input_file, target_file, num=10000000):
        self.inputs = np.load(input_file)[:num]
        self.targets = np.load(target_file)[:num]
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_vector = self.inputs[idx]
        target_vector = self.targets[idx]
        return torch.tensor(input_vector, dtype=torch.float32), torch.tensor(target_vector, dtype=torch.float32)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# for model_from in model_set:
for model_from in model_set:
    for model_to in model_set:
        if model_from == model_to:
            continue
        
        print(f"{model_from} --TO-- {model_to} ")

        file_path = "embeddings/scifact/" + model_from + "_all.npy"
        data = np.load(file_path)
        input_size = data.shape[-1]
        file_path = "embeddings/scifact/" + model_to + "_all.npy"
        data = np.load(file_path)
        output_size = data.shape[-1]

        directory = "test_models/" + model_from + '--TO--' + model_to
        if not os.path.exists(directory):
            os.makedirs(directory)
        print('load dataset start')
        print(f"training on {dataset_name}")
        try:
            vectors_from = '/home/work/mintaek2/embeddings/'+dataset_name +'/' + model_from + '_combine.npy'
            vectors_to = '/home/work/mintaek2/embeddings/' +dataset_name +'/' + model_to + '_combine.npy'
            combined_dataset = VectorDataset(vectors_from, vectors_to)
            print("combined train set")
        except:

            vectors_from = 'embeddings/'+dataset_name +'/' + model_from + '_all.npy'
            vectors_to = 'embeddings/' +dataset_name +'/' + model_to + '_all.npy'
            combined_dataset = VectorDataset(vectors_from, vectors_to)
            print(f"fine tune dataset: {dataset_name}")
        
        
        val_size = int(0.001 * len(combined_dataset))  # 0.1% for validation
        train_size = len(combined_dataset) - val_size
        print(f"=========== {train_size} ==============")
        train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        print(f"Train dataset size: {len(train_dataset)}, Train loader batches: {len(train_loader)}")
        print(f"val dataset size:   {len(val_dataset)},    Val  loader batches: {len(val_loader)}")        
        print("dataset ready")
        mlp_model = Big_MLP(input_size, output_size).to(device)
        optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)

        best_val_loss = float('inf')
        ppath = dataset_name + '_big_model.pth'
        best_model_path = os.path.join(directory, ppath)
        print("model ready: ready to train")
        for epoch in range(num_epochs):
            mlp_model.train()
            running_train_loss = 0.0
            
            
            for inputs, targets in tqdm(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = mlp_model(inputs)
                loss = fuzzy_loss(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
                # print(f"present loss : {running_train_loss}")

            avg_train_loss = running_train_loss / len(train_loader)

            # Validation
            mlp_model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = mlp_model(inputs)
                    loss = fuzzy_loss(outputs, targets)
                    running_val_loss += loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            print(f"avg val loss : {avg_val_loss}")
            # Save the model with the lowest validation loss
            print(f"epoch: {epoch}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(mlp_model.state_dict(), best_model_path)
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")

        print(f"Best model saved at {best_model_path}")


        