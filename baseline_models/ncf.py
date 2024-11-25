from torch_geometric.datasets import MovieLens1M
from torch_geometric.datasets import MovieLens100K
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import pickle
from torch_geometric.loader import NeighborLoader
import numpy as np
from torch.utils.data import Dataset, DataLoader

from torch_geometric.utils import negative_sampling, subgraph, to_networkx

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


class NCF(torch.nn.Module):
    """ Neural Collaborative Filtering (NCF)

        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the movie ratings for training
            all_movieIds (list): List containing all movieIds (train + test)
    """

    def __init__(self, num_users, num_items):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=32)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=32)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)

    def forward(self, user_input, item_input):

        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # For training uncomment below

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred

        # return user_embedded, item_embedded


ncf_model = NCF(user_features.shape[0], movie_features.shape[0])

ncf_model = ncf_model.to(device)

def data_creation(edge_index, all_data_edge_index, num_neg_samples, all_items):
  user_data = []
  item_data = []
  label_data = []

  neg_user_data = []
  neg_item_data = []
  neg_label_data = []

  for i in range(edge_index.shape[1]):
    # print(edge_index[:, i])
    user_data.append(edge_index[0][i].item())
    item_data.append(edge_index[1][i].item())
    label_data.append(1)

    # import pdb; pdb.set_trace()

  print('user data len : ',len(user_data))

  num_user = edge_index[0].unique()


  all_pos_user_item_set = set()

  # import pdb; pdb.set_trace()

  for i in range(all_data_edge_index.shape[1]):
    curr_user = all_data_edge_index[0][i].item()
    curr_item = all_data_edge_index[1][i].item()
    all_pos_user_item_set.add((curr_user, curr_item))

    # import pdb; pdb.set_trace()

  # import pdb; pdb.set_trace()
  for curr_user in num_user:
    for j in range(num_neg_samples):
      negative_item = np.random.choice(all_items)
      while (curr_user.item(), negative_item.item()) in all_pos_user_item_set:
            negative_item = np.random.choice(all_items)
      neg_user_data.append(curr_user.item())
      neg_item_data.append(negative_item.item())
      neg_label_data.append(0)
      # import pdb; pdb.set_trace()

  print('neg user data len : ',len(neg_user_data))

  user_data.extend(neg_user_data)
  item_data.extend(neg_item_data)
  label_data.extend(neg_label_data)
  assert len(neg_user_data) == num_neg_samples * len(num_user)
  return torch.tensor(user_data), torch.tensor(item_data), torch.tensor(label_data)

all_items = data_edge_index[1].unique()

train_user_data, train_item_data, train_label_data =  data_creation(edge_index = train_edge_index, all_data_edge_index = data_edge_index, num_neg_samples = 70, all_items = all_items)
val_user_data,val_item_data, val_label_data =  data_creation(edge_index = val_edge_index, all_data_edge_index = data_edge_index, num_neg_samples = 70, all_items = all_items)

class CustomDataset3D(Dataset):
    def __init__(self, X1, X2, X3):
        self.X1 =  X1
        self.X2 = X2
        self.X3 = X3

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, index):
        return self.X1[index], self.X2[index], self.X3[index]
    
train_dataloader = DataLoader(CustomDataset3D(train_user_data, train_item_data, train_label_data), batch_size=1024, shuffle = True)
val_dataloader = DataLoader(CustomDataset3D(val_user_data, val_item_data, val_label_data), batch_size=1024, shuffle = True)

optimizer = torch.optim.Adam(ncf_model.parameters())

loss_fn = nn.BCELoss()


def val(ncf_model, val_dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ncf_model.to(device)

    ncf_model.eval()
    with torch.no_grad():
        epoch_loss = 0.0
        for batch in val_dataloader:
          user_data = batch[0].to(device)
          item_data = batch[1].to(device)
          label_data = batch[2].to(device)

          pred = ncf_model(user_data, item_data)
          loss = loss_fn(pred, label_data.view(-1,1).float())

          epoch_loss += loss


        print('val epoch loss :', epoch_loss.item())
        print('\n')



def train(ncf_model, train_dataloader, val_dataloader, optimizer):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ncf_model.to(device)

    ncf_model.train()

    min_epoch_loss = 1e9
    for epoch in tqdm(range(10)):
      epoch_loss = 0.0
      for batch in train_dataloader:
        user_data = batch[0].to(device)
        item_data = batch[1].to(device)
        label_data = batch[2].to(device)

        
        pred = ncf_model(user_data, item_data)

        # import pdb; pdb.set_trace()
        loss = loss_fn(pred, label_data.view(-1,1).float())

        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


      print('train epoch loss : ', epoch_loss)
      print('\n')
      val(ncf_model, val_dataloader)
      torch.save(ncf_model.state_dict(), '/content/drive/MyDrive/Colab Notebooks/ncf_model.pt')


train(ncf_model, train_dataloader, val_dataloader, optimizer )

class NCF_inference(torch.nn.Module):
    """ Neural Collaborative Filtering (NCF)

        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the movie ratings for training
            all_movieIds (list): List containing all movieIds (train + test)
    """

    def __init__(self, num_users, num_items):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=32)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=32)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)

    def forward(self, user_input, item_input):

        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # # For training uncomment below

        # # Concat the two embedding layers
        # vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # # Pass through dense layer
        # vector = nn.ReLU()(self.fc1(vector))
        # vector = nn.ReLU()(self.fc2(vector))

        # # Output layer
        # pred = nn.Sigmoid()(self.output(vector))

        # return pred

        return user_embedded, item_embedded


ncf_model_inference = NCF_inference(user_features.shape[0], movie_features.shape[0])

ncf_model_inference.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/ncf_model.pt', weights_only=True), strict=True)

ncf_model_inference = ncf_model_inference.to(device)

test_user = test_edge_index[0].unique()
test_item = test_edge_index[1].unique()

print(test_user.shape)
print(test_item.shape)
user_test_embedding, item_test_embedding = ncf_model_inference(test_user.to(device), test_item)

test_user_map = {key.item(): i for i,key in enumerate(test_user)}
test_item_map = {key.item(): i for i, key in enumerate(test_item)}

p1, p5, p10, mrr = [], [], [], []
r10, map_scores, ndcg_scores = [], [], []

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
# itm = embeddings[user_features.shape[0]:]
users = test_user

for i in tqdm(range(len(users))):
  curr_node = users[i]
  self_emb = user_test_embedding[i]
  # self_emb = embeddings[curr_node]
  similarities = cos(self_emb, item_test_embedding)
  sorted_indices = torch.argsort(similarities, descending=True)
  sorted_indices = [test_item[i] for i in sorted_indices]
  test_edges_q_indices = test_edge_index[1][test_edge_index[0] == curr_node]

  # import pdb; pdb.set_trace()

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