import numpy as np
import os
from tqdm import tqdm
import torch

adj_matrix_base_path = "./category_adjacency_matrix"
category_vectors_path = "./category_vectors.npy"
word2vec_path = "./w2v_16d.npy"

w2v = np.load(word2vec_path)  

category_vectors = np.load(category_vectors_path) 

adj_matrix_files = []
for batch in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]:
    folder_path = os.path.join(adj_matrix_base_path, str(batch))
    if batch < 8000:
        parts = range(8)  
    else:
        parts = range(2) 
    for part in parts:
        file_path = os.path.join(folder_path, f"Processed_AdjacencyMatrix_batch_{batch}_part_{part}.npy")
        if os.path.exists(file_path):
            adj_matrix_files.append(file_path)

total_images = category_vectors.shape[0]
word_num = w2v.shape[0]
word_length = w2v.shape[1]
result_matrix = np.zeros((total_images, word_num * word_length), dtype=np.float32)

current_idx = 0

for file_path in tqdm(adj_matrix_files, desc="Processing adjacency matrix files"):
    adj_matrices = np.load(file_path)  
    
    for adj_matrix in tqdm(adj_matrices, desc="Processing matrices in file", leave=False):
        cat_vector = category_vectors[current_idx] 
        
        weighted_matrix = np.zeros((word_num, word_length), dtype=np.float32)
        for i in range(word_num):
            weighted_matrix[i] = cat_vector[i] * w2v[i] 
        
        result = np.matmul(adj_matrix, weighted_matrix) 
        
        result_flat = result.flatten()
        
        result_matrix[current_idx] = result_flat
        
        current_idx += 1

print("总图像数为", current_idx)

output_path = f"image_representation_57396_{word_length}.pt"  
result_tensor = torch.from_numpy(result_matrix) 
torch.save(result_tensor, output_path) 

print(f"Processing complete. Result saved to {output_path}")
print(f"Result matrix shape: {result_matrix.shape}")