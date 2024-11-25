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
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

PATH = "path to save model"

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

loss_name = "bpr"
model_type = "GCN"

movie_lens = MovieLens100K('./data/movie_lens')[0]
movie_features = movie_lens["movie"]["x"]
user_features = movie_lens["user"]["x"]
data = movie_lens[("user", "rates", "movie")]
mask = data["rating"] >= 3
data_edge_index = data["edge_index"][:, mask]
data_edge_label = data["rating"][mask]

user_num_nodes = user_features.shape[0]
train_nodes, testing_nodes = train_test_split(range(user_num_nodes), test_size=0.2, random_state=42)
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
    

def bpr_loss(all_embeds, pos_edge_index, num_neg_samples=50):
    loss = 0.0
    visited_nodes = set()

    for i in range(pos_edge_index.shape[1]):
        u, v = pos_edge_index[:, i]  # Extract a positive edge (u -> v)
        if u in visited_nodes:
            continue
        visited_nodes.add(u)

        u_embedding = all_embeds[u].unsqueeze(0)
        pos_nbr_embedding = all_embeds[pos_edge_index[1][pos_edge_index[0] == u]]

        neg_nbors_indices = sample_neg_edges(u, pos_edge_index, num_sample=1)
        neg_nbrs = all_embeds[neg_nbors_indices]

        pos_score = cos(pos_nbr_embedding, u_embedding) 
        neg_scores = cos(neg_nbrs, u_embedding)  

        # BPR loss: log-sigmoid of the score differences
        bpr_terms = -F.logsigmoid(pos_score - neg_scores)  
        loss += bpr_terms.sum()

    return loss/pos_edge_index.shape[1]

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
def margin_loss(all_embeds, pos_edge_index, margin):
    loss = 0.0
    visited_nodes = set()
    for i in range(pos_edge_index.shape[1]):
        u, v = pos_edge_index[:, i]
        if u in visited_nodes:
            continue
        visited_nodes.add(u)
        u_embedding = all_embeds[u].unsqueeze(0)

        neg_nbors_indices = sample_neg_edges(u, pos_edge_index, num_sample = 15)

        # Positive neighbors - all v where u -> v
        pos_nbrs = all_embeds[pos_edge_index[1][pos_edge_index[0] == u]]

        # Negative neighbors - all v where u->v in negative sampling
        neg_nbrs = all_embeds[neg_nbors_indices]

        pos_scores = cos(pos_nbrs, u_embedding)  
        neg_scores = cos(neg_nbrs, u_embedding)  

        len_pos = pos_scores.shape[0]
        len_neg = neg_scores.shape[0]

        expanded_pos_scores = pos_scores.unsqueeze(1).expand(len_pos, len_neg)
        expanded_neg_scores = neg_scores.unsqueeze(0).expand(len_pos, len_neg)
        import pdb; pdb.set_trace()
        # Margin-based loss calculation with max function
        margin_penalty = torch.max(margin + expanded_neg_scores - expanded_pos_scores, torch.tensor(0.0, device=all_embeds.device))
        loss += margin_penalty.sum()

    return loss/pos_edge_index.shape[1]

model = GNN(model_type, in_channels= 32, hidden_channels=128, out_channels=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model=model.to(device)

def sample_neg_edges(curr_node, edge_index, num_sample):
  mask = torch.ones(edge_index[1].shape[0], dtype = torch.bool)
  pos_items = edge_index[1][edge_index[0] == curr_node]
  mask[pos_items] = False
  assert mask.shape == edge_index[1].shape
  val = edge_index[1][mask][:num_sample]
  return val

def negative_sample_calculation(nodes, edge_index):
  neg_sample_dict = defaultdict()
  for i in tqdm(range(len(nodes))):
    neg_val = sample_neg_edges(nodes[i], edge_index, num_sample=1000)
    neg_sample_dict[nodes[i]] = neg_val

  return neg_sample_dict


min_ep_loss = 1e6
num_nodes = train_edge_index.max().item() + 1
EPOCHS = 1

for i in tqdm(range(EPOCHS)):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    embeddings = model(user_features, movie_features, train_edge_index)
    if loss_name == "bpr":
        loss = bpr_loss(embeddings, train_edge_index)
    if loss_name == "margin_loss":
        loss = margin_loss(embeddings, train_edge_index, margin = 0.01)

    if loss < min_ep_loss:

        min_ep_loss = loss
        torch.save(model.state_dict(), PATH)
    loss.backward()
    optimizer.step()

model.load_state_dict(torch.load(PATH))

model.eval()
with torch.no_grad():
  embeddings = model(user_features, movie_features, test_edge_index)
users = test_edge_index[0].unique()
p1, p5, p10, mrr = [], [], [], []
r10, map_scores, ndcg_scores = [], [], []

itm = embeddings[user_features.shape[0]:]
users = test_edge_index[0].unique()

for i in tqdm(range(len(users))):
  curr_node = users[i]
  self_emb = embeddings[curr_node]
  similarities = cos(self_emb, itm)
  sorted_indices = torch.argsort(similarities, descending=True)
  sorted_indices = [i + user_features.shape[0] for i in sorted_indices]
  test_edges_q_indices = test_edge_index[1][test_edge_index[0] == curr_node]

  top_k = sorted_indices[:10]
  p_10 =  sum(1 for i in top_k if i in test_edges_q_indices)
  # pdb.set_trace()
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
  # if p_10 > 0:
  #   print("user ", curr_node)
  #   print( top_k)
  #   print("p10 ", p_10)
  #   break

  avg_precision = 0.0
  num_relevant = 0
  for rank, node in enumerate(top_k, start=1):
      if node in test_edges_q_indices:
          num_relevant += 1
          avg_precision += num_relevant / rank
  if num_relevant > 0:
      map_scores.append(avg_precision / len(test_edges_q_indices))
  else:
      map_scores.append(0.0)

  # NDCG Calculation
  dcg = 0.0
  for rank, node in enumerate(top_k, start=1):
      if node in test_edges_q_indices:
          dcg += 1 / np.log2(rank + 1)
  # Compute IDCG (ideal DCG)
  ideal_relevant = min(len(test_edges_q_indices), 10)
  idcg = sum(1 / np.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
  ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)


# Final Metrics
print("Precision@1:", np.mean(p1))
print("Precision@5:", np.mean(p5))
print("Precision@10:", np.mean(p10))
print("Recall@10:", np.mean(r10))
print("MRR:", np.mean(mrr))
print("MAP:", np.mean(map_scores))
print("NDCG@10:", np.mean(ndcg_scores))