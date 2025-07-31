import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from collections import Counter
from tqdm import tqdm
import os

class Word2VecDataset(Dataset):
    def __init__(self, pairs, word_freq, num_neg_samples, vocab_size):
        self.pairs = pairs
        self.word_freq = word_freq
        self.num_neg_samples = num_neg_samples
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target, context = self.pairs[idx]
        neg_samples = np.random.choice(self.vocab_size, self.num_neg_samples, p = self.word_freq) 
        return torch.tensor(target), torch.tensor(context), torch.tensor(neg_samples)

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim): 
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target, context, neg_samples): 
        target_emb = self.target_embeddings(target) 
        context_emb = self.context_embeddings(context)
        pos_score = torch.sum(target_emb * context_emb, dim=1) 
        pos_loss = -torch.log(torch.sigmoid(pos_score)).mean() 
        neg_emb = self.context_embeddings(neg_samples)
        neg_score = torch.bmm(neg_emb, target_emb.unsqueeze(2)).squeeze(2) 
        neg_loss = -torch.log(torch.sigmoid(-neg_score)).mean()
        return pos_loss + neg_loss 

class EarlyStopper:
    def __init__(self, patience = 6, min_delta = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf') 

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

if __name__ == "__main__":
    tokenized_sentences = [] 
    a = np.load('category_vectors.npy') 
    print("Shape of a:", a.shape)

    vocab_size = a.shape[1] 
    num_neg_samples = 5 
    skipped_sentences = 0  
    batch_size = 512

    for i in range(a.shape[0]): 
        t = []
        for j in range(a.shape[1]): 
            if a[i][j] != 0:
                t.append(j) 
        if len(t) < 2:
            skipped_sentences += 1
            continue
        tokenized_sentences.append(t)
    corpus = tokenized_sentences
    print("First sentence:", corpus[0])
    print("Number of sentences with fewer than 2 words (skipped):", skipped_sentences)

    word_counts = Counter([word for sentence in corpus for word in sentence]) 
    word_freq = np.array([word_counts.get(i, 0) for i in range(vocab_size)], dtype=np.float32)
    word_freq = word_freq ** (3 / 4) 
    word_freq = word_freq / (word_freq.sum() + 1e-10)  

    pairs = []
    for sentence in corpus:
        n = len(sentence) 
        for i in range(n):
            target = sentence[i]
            contexts = sentence[:i] + sentence[i + 1:] 
            for context in contexts:
                pairs.append((target, context))

    val_ratio = 0.2
    split_idx = int(len(pairs) * (1 - val_ratio))
    train_pairs, val_pairs = random_split(pairs, [split_idx, len(pairs) - split_idx])

    train_dataset = Word2VecDataset(train_pairs, word_freq, num_neg_samples, vocab_size)
    val_dataset = Word2VecDataset(val_pairs, word_freq, num_neg_samples, vocab_size)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True) 
    val_loader = DataLoader(val_dataset, batch_size = batch_size)

    embedding_dims = [16, 24, 32]

    for embedding_dim in embedding_dims:
        print(f"\nTraining with embedding_dim={embedding_dim}\n{'='*40}")

        # 训练配置
        model = Word2Vec(vocab_size = 48, embedding_dim = embedding_dim)
        optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-5)
        early_stopper = EarlyStopper(patience = 6, min_delta = 0.001)
        max_epochs = 100

        # 训练循环
        for epoch in range(max_epochs):
            model.train()
            train_loss = 0
            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [Train, dim={embedding_dim}]")
            for targets, contexts, neg_samples in train_loader_tqdm:
                optimizer.zero_grad()
                loss = model(targets, contexts, neg_samples)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{max_epochs} [Val, dim={embedding_dim}]")
            with torch.no_grad():
                for targets, contexts, neg_samples in val_loader_tqdm:
                    loss = model(targets, contexts, neg_samples)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch + 1}/{max_epochs}")
            print(f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            if early_stopper.early_stop(avg_val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        embeddings = model.target_embeddings.weight.data 
        print("Embedding shape:", embeddings.shape)
        numpy_array = embeddings.numpy()
        os.makedirs("../result", exist_ok=True)
        embedding_save_path = f'../result/w2v_{embedding_dim}d.npy'
        np.save(embedding_save_path, numpy_array)
        print(f"Embeddings saved to {embedding_save_path}")

        # model_save_path = f'../result/w2v_model_{embedding_dim}d.pth'
        # torch.save(model.state_dict(), model_save_path)
        # print(f"Model parameters saved to {model_save_path}")