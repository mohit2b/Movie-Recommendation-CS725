from torch_geometric.datasets import MovieLens100K
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
import random
import time

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

PATH = "path to load the model"
model_type = "GCN"

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)


movie_lens = MovieLens100K('./data/movie_lens')[0]
movie_features = movie_lens["movie"]["x"]
user_features = movie_lens["user"]["x"]
data = movie_lens[("user", "rates", "movie")]
mask = data["rating"] >= 3
data_edge_index = data["edge_index"][:, mask]
data_edge_label = data["rating"][mask]

user_num_nodes = user_features.shape[0]
train_nodes, testing_nodes = train_test_split(range(user_num_nodes), test_size=0.2, random_state=seed)
val_nodes, test_nodes = testing_nodes[:len(testing_nodes)//2], testing_nodes[len(testing_nodes)//2: ]

Y = data_edge_index[0]
train_mask = torch.isin(Y, torch.tensor(train_nodes))
train_edge_index = data_edge_index[:, train_mask]
val_mask = torch.isin(Y, torch.tensor(val_nodes))
val_edge_index = data_edge_index[:, val_mask]
test_mask = torch.isin(Y, torch.tensor(test_nodes))
test_edge_index = data_edge_index[:, test_mask]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

user_features = user_features.to(device)
movie_features = movie_features.to(device)

train_edge_index = train_edge_index.to(device)
val_edge_index = val_edge_index.to(device)
test_edge_index = test_edge_index.to(device)

train_edge_index[1] += (user_features.shape[0])
val_edge_index[1] += (user_features.shape[0])
test_edge_index[1] += (user_features.shape[0])

class GNN(torch.nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.model_type = model_type
        self.fc1 = nn.Linear(user_features.shape[1], in_channels)
        self.fc2 = nn.Linear(movie_features.shape[1], in_channels)
        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            self.bn1= torch.nn.BatchNorm1d(hidden_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=2, concat=True)
            self.conv2 = GATConv(hidden_channels * 2, out_channels, heads=2, concat=True)
            self.bn1= torch.nn.BatchNorm1d(hidden_channels*2)

    def forward(self, x, y, edge_index):
        x = self.fc1(x)
        y = self.fc2(y)
        z = torch.cat((x, y), dim=0)
        z = F.relu(self.bn1(self.conv1(z, edge_index)))
        z = self.conv2(z, edge_index)
        return z
    
model = GNN(model_type, in_channels= 32, hidden_channels=128, out_channels=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model=model.to(device)
model.load_state_dict(torch.load(PATH))

model.eval()
with torch.no_grad():
  embeddings = model(user_features, movie_features, test_edge_index)

class RandomProjection:
    counter = 0
    non_prior_b  = 0
    def __init__(self, device, nplanes, feature_dim, test_users, test_features, test_edge_index):
        self.device = device
        self.nbits = nplanes
        self.dplane = feature_dim
        self.hyperplanes = torch.rand(nplanes, feature_dim, device=device) - 0.5
        # print(f"Initialized {self.hyperplanes.shape} hyperplane normal vectors.")
        self.buckets = defaultdict(list)
        self.test_features = test_features
        self.user_embeddings =  test_features[:test_users]
        self.item_embeddings =  test_features[test_users:]

        test_hash_codes = self.get_hash_code(test_features)
        self.user_hash_codes = test_hash_codes[:test_users]
        self.movie_hash_codes = test_hash_codes[test_users:]

        # self.test_labels = test_labels
        self.bucketize()
        self.cosine = torch.nn.CosineSimilarity(dim=1)
        self.test_edge_index = test_edge_index

    def get_hash_code(self, features):
        hash_codes =  torch.matmul(features, self.hyperplanes.T) > 0
        return hash_codes.int()

    def bucketize(self):
        for idx, hash_code in enumerate(self.movie_hash_codes):
            hash_code_tuple = tuple(hash_code.tolist())
            self.buckets[hash_code_tuple].append(idx)
        # print(f"Generated and bucketized hash codes for training features with {len(self.buckets)} buckets.")

    def get_topk(self, min_candidates):
        # print(f"find the nearest neighbors using the bucketized hash codes.")
        results = []
        bucket_hashes = torch.tensor(list(self.buckets.keys()), device=self.device)
        min_buckets = len(self.buckets.keys())
        test_users = self.test_edge_index[0].unique()
        for user in test_users:
          user_hash_code = self.user_hash_codes[user]
          distances = torch.sum(bucket_hashes != user_hash_code, dim=1)
          sorted_bindices = torch.argsort(distances)
          candidate_indices = []
          bid = 0
          for idx in sorted_bindices:
              bucket_hash = tuple(bucket_hashes[idx].cpu().numpy())
              candidate_indices.extend(self.buckets[bucket_hash])
              if len(candidate_indices) >= min_candidates:
                  if bid > 0:
                      #underflow
                      self.non_prior_b += 1
                      candidate_indices = candidate_indices[:min_candidates]
                  break
              bid += 1
          results.append(candidate_indices)
        return results

    def evaluate(self, query_candidates):
      p1, p5, p10, mrr = [], [], [], []
      r10, map_scores, ndcg_scores = [], [], []

      test_users = self.test_edge_index[0].unique()
      for idx, indices in tqdm(enumerate(query_candidates)):
        curr_node = test_users[idx]
        self_emb = self.user_embeddings[curr_node]
        similarities = cos(self_emb, self.item_embeddings[indices])
        sorted_indices = torch.argsort(similarities, descending=True)

        sorted_indices = [indices[i] for i in sorted_indices]

        sorted_indices = [i + user_features.shape[0] for i in sorted_indices]
        test_edges_q_indices = self.test_edge_index[1][self.test_edge_index[0] == curr_node]
        top_k = sorted_indices[:10]
        p_10 =  sum(1 for i in top_k if i in test_edges_q_indices)
        p10.append(p_10 / 10)
        p_5 =  sum(1 for i in top_k[:5] if i in test_edges_q_indices)
        p5.append(p_5 / 5)
        p_1 =  1 if top_k[0] in test_edges_q_indices else 0
        p1.append(p_1 / 1)
        r10.append(p_10/len(test_edges_q_indices))
        for rank, node in enumerate(top_k):
            if node in test_edges_q_indices:
                mrr.append(1 / (rank + 1))
                break

        dcg = 0.0
        for rank, node in enumerate(top_k, start=1):
            if node in test_edges_q_indices:
                dcg += 1 / np.log2(rank + 1)
        # Compute IDCG (ideal DCG)
        ideal_relevant = min(len(test_edges_q_indices), 10)
        idcg = sum(1 / np.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
      print("Precision@1:", np.mean(p1))
      print("Precision@5:", np.mean(p5))
      print("Precision@10:", np.mean(p10))
      print("Recall@10:", np.mean(r10))
      print("MRR:", np.mean(mrr))
      print("NDCG@10:", np.mean(ndcg_scores))
      return p10
    
feature_dim = embeddings.shape[1]
test_users = user_features.shape[0]

for nplanes in range(1,7):
  rp = RandomProjection(device, nplanes, feature_dim, test_users, embeddings, test_edge_index)
  s =  time.time()
  results = rp.get_topk(min_candidates=100)
  print(f"-------------- nplanes: {nplanes} -----------")
  p10 = rp.evaluate(results)
  e =  time.time()
  print("time ", e-s)
  for key, value in rp.buckets.items():
    print(key, " - ", len(value))
  print(f"------------------------")


