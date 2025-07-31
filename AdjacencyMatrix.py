import numpy as np
import torch
from PIL import Image
import networkx as nx
from collections import Counter
import pandas as pd
from tqdm import tqdm
import os

class StreetView_Graph:
    def __init__(self):
        self.svi_seg = None
        self.adjacency_dict = None
        self.adjacency_matrix = None
        self.weight_graph = None

    def calculate_neighborhood(self):
        if self.svi_seg is None:
            raise ValueError("svi_seg is not initialized. Please provide a segmented image array.")
        unique_labels = np.unique(self.svi_seg)
        self.adjacency_dict = {label: [] for label in unique_labels}

        for i in range(self.svi_seg.shape[0]):
            for j in range(self.svi_seg.shape[1]):
                label = self.svi_seg[i, j]
                neighbors = []
                if i > 0 and self.svi_seg[i - 1, j] != label:
                    neighbors.append(self.svi_seg[i - 1, j])
                if i < self.svi_seg.shape[0] - 1 and self.svi_seg[i + 1, j] != label:
                    neighbors.append(self.svi_seg[i + 1, j])
                if j > 0 and self.svi_seg[i, j - 1] != label:
                    neighbors.append(self.svi_seg[i, j - 1])
                if j < self.svi_seg.shape[1] - 1 and self.svi_seg[i, j + 1] != label:
                    neighbors.append(self.svi_seg[i, j + 1])
                self.adjacency_dict[label].extend(neighbors)

        return self.adjacency_dict

    def generate_adjacency_matrix(self):
        if self.adjacency_dict is None:
            self.calculate_neighborhood()

        self.adjacency_matrix = pd.DataFrame()
        for i, cate in enumerate(self.adjacency_dict.keys()):
            if i == 0:
                self.adjacency_matrix = pd.DataFrame(Counter(self.adjacency_dict[cate]), index=[cate])
            else:
                self.adjacency_matrix = pd.concat(
                    [self.adjacency_matrix, pd.DataFrame(Counter(self.adjacency_dict[cate]), index=[cate])])

        self.adjacency_matrix.fillna(0, inplace=True)

        # Filter out less significant categories (e.g., less than 1% of total pixels), 因为对角线为0, 所以这个分别按行和列过滤没问题
        row_sums = self.adjacency_matrix.sum(axis=1)
        total_sum = row_sums.sum()
        mask = (row_sums / total_sum) < 0.01
        self.adjacency_matrix.loc[mask, :] = 0

        col_mask = self.adjacency_matrix.T.sum(axis=1) / total_sum < 0.01
        self.adjacency_matrix.loc[:, col_mask] = 0

        shape = (150, 150)
        zero_lst = [[0 for _ in range(shape[1])] for _ in range(shape[0])]
        arr = np.array(zero_lst)
        unique_labels = np.unique(self.svi_seg)
        if len(unique_labels) >= 2:
            for i in unique_labels:
                for j in unique_labels:
                    arr[int(i)][int(j)] = self.adjacency_matrix.loc[i, j]
        for i in range(len(arr)):
            arr[i][i] = arr[i][i] + 1

        return arr

def calculate_adjacency_and_embedding(a):
    graph = StreetView_Graph()
    graph.svi_seg = np.array(a)
    graph.calculate_neighborhood()
    em = graph.generate_adjacency_matrix()
    return em

if __name__ == "__main__":
    base_dir = "./split"
    output_base_dir = os.path.join(os.getcwd(), "result_split")
    os.makedirs(output_base_dir, exist_ok=True)
    batch_numbers = range(1000, 9000, 1000)

    for batch_num in tqdm(batch_numbers, desc="Processing batches"):
        # Create output subdirectory for this batch
        output_dir = os.path.join(output_base_dir, str(batch_num))
        os.makedirs(output_dir, exist_ok=True)

        batch_dir = os.path.join(base_dir, str(batch_num))

        # Determine number of parts (special case for 8000)
        if batch_num == 8000:
            num_parts = 2  # mask_batch_8000_epoch_0_part_0.npy to part_2.npy
        else:
            num_parts = 8  # Each batch has batch_num/1000 parts

        for part_idx in tqdm(range(num_parts), desc=f"Processing batch {batch_num} parts"):
            file_path = os.path.join(batch_dir, f"mask_batch_{batch_num}_epoch_0_part_{part_idx}.npy")

            loaded_array = np.load(file_path)
            a = loaded_array.tolist()  # Convert to list for processing

            # Process each image in the part
            res = None
            for j in tqdm(range(len(a)), desc=f"Computing adjacency for batch {batch_num} part {part_idx}"):
                t = a[j]
                embedding = calculate_adjacency_and_embedding(t)
                arr1_expanded = embedding[np.newaxis, :, :]

                if res is None:
                    res = arr1_expanded
                else:
                    res = np.concatenate((res, arr1_expanded), axis=0)

            # Save result for this part in the corresponding batch subdirectory
            output_file = os.path.join(output_dir, f"AdjacencyMatrix_batch_{batch_num}_part_{part_idx}.npy")
            np.save(output_file, res)
            print(f"Saved adjacency matrix for batch {batch_num} part {part_idx} to {output_file} with shape {res.shape}")