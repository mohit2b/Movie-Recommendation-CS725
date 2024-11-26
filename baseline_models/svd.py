from torch_geometric.datasets import MovieLens100K
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

movie_lens = MovieLens100K('./data/movie_lens')[0]

movie_features = movie_lens["movie"]["x"]
user_features = movie_lens["user"]["x"]

data = movie_lens[("user", "rates", "movie")]
mask = data["rating"] >= 3

data_edge_index = data["edge_index"][:, mask]
data_edge_label = data["rating"][mask]

user_num_nodes = user_features.shape[0]

#Split 80/10/10 train val test
train_nodes, testing_nodes = train_test_split(range(user_num_nodes), test_size=0.2, random_state=42)
val_nodes, test_nodes = testing_nodes[:len(testing_nodes)//2], testing_nodes[len(testing_nodes)//2: ]

# Extract user and movie indices
user_indices = data_edge_index[0].numpy()
movie_indices = data_edge_index[1].numpy()

# Extract ratings as values
ratings = data_edge_label.numpy()

# Create a sparse user-item interaction matrix
num_users = user_features.shape[0]
num_movies = movie_features.shape[0]

interaction_matrix = coo_matrix(
    (ratings, (user_indices, movie_indices)),
    shape=(num_users, num_movies)
)

interaction_matrix.shape

k = 16  # Number of latent factors
interaction_matrix = interaction_matrix.astype(np.float32)

# Perform SVD using scipy
U, sigma, Vt = svds(interaction_matrix, k=k)

U = torch.tensor(U, dtype=torch.float32)
V = torch.tensor(Vt.astype(np.float32), dtype=torch.float32)
sigma = torch.tensor(sigma.astype(np.float32), dtype=torch.float32)
Sigma = torch.diag(sigma)

# Reconstruct the interaction matrix
R_pred = torch.tensor(U @ Sigma @ V)
R_pred = torch.round(R_pred).clamp(0, 5).int()

# Convert train, val, and test nodes to masks
train_mask = torch.isin(torch.tensor(user_indices), torch.tensor(train_nodes))
val_mask = torch.isin(torch.tensor(user_indices), torch.tensor(val_nodes))
test_mask = torch.isin(torch.tensor(user_indices), torch.tensor(test_nodes))

# Create sparse matrices for train, validation, and test
train_matrix = coo_matrix(
    (ratings[train_mask], (user_indices[train_mask], movie_indices[train_mask])),
    shape=(num_users, num_movies)
)

val_matrix = coo_matrix(
    (ratings[val_mask], (user_indices[val_mask], movie_indices[val_mask])),
    shape=(num_users, num_movies)
)

test_matrix = coo_matrix(
    (ratings[test_mask], (user_indices[test_mask], movie_indices[test_mask])),
    shape=(num_users, num_movies)
)

# Get predicted ratings for the test set
test_users, test_movies = test_matrix.nonzero()
test_ratings_actual = test_matrix.data
test_ratings_pred = R_pred[test_users, test_movies].numpy()

# Compute RMSE
rmse = np.sqrt(np.mean((test_ratings_actual - test_ratings_pred) ** 2))
print(f"Test RMSE: {rmse:.4f}")

# Metrics Implementation
# Top-k Precision and Recall functions (Exact match of both movie index and rating)
def top_k_precision(predicted_indices, predicted_ratings, relevant_indices, relevant_ratings, k):
  precisions = []
  # Looping over pred_ind, rel_ind, pred_ratings, rel_ratings for each user
  for pred_ind, rel_ind, pred_ratings, rel_ratings in zip(predicted_indices, relevant_indices, predicted_ratings, relevant_ratings):
    # Count relevant items in the top k predictions with exact rating match
    relevant_items_in_top_k = 0
    num_relevant = len(rel_ind)
    if num_relevant == 0:
        continue
    for idx in pred_ind:
        if idx in rel_ind:
            pred_rating = pred_ratings[np.where(pred_ind == idx)[0][0]]
            rel_rating = rel_ratings[np.where(rel_ind == idx)[0][0]]
            if pred_rating == rel_rating and pred_rating >= 3:
                relevant_items_in_top_k += 1
    precisions.append(relevant_items_in_top_k / k)
  return np.mean(precisions)

def top_k_recall(predicted_indices, predicted_ratings, relevant_indices, relevant_ratings, k):
  recalls = []
  for pred_ind, rel_ind, pred_ratings, rel_ratings in zip(predicted_indices, relevant_indices, predicted_ratings, relevant_ratings):
    # Count how many relevant items are in the top k predictions with exact rating match
    relevant_items_in_top_k = 0
    num_relevant = len(rel_ind)
    if num_relevant == 0:
        continue
    for idx in pred_ind:
        if idx in rel_ind:
            pred_rating = pred_ratings[np.where(pred_ind == idx)[0][0]]
            rel_rating = rel_ratings[np.where(rel_ind == idx)[0][0]]
            if pred_rating == rel_rating and pred_rating >= 3:
                relevant_items_in_top_k += 1
    recalls.append(relevant_items_in_top_k / num_relevant)
  return np.mean(recalls)

def mean_reciprocal_rank(predicted_indices, relevant_indices):
    reciprocal_ranks = []
    for pred, rel in zip(predicted_indices, relevant_indices):
        num_relevant = len(rel)
        if num_relevant == 0:
          continue
        match_indices = [i for i, p in enumerate(pred) if p in rel]
        if match_indices:
            reciprocal_ranks.append(1 / (match_indices[0] + 1))  # Rank is 1-based
        else:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks)

def ndcg(predicted_indices, relevant_indices, relevant_scores, k):
    def dcg(scores, k):
        return np.sum([
            score / np.log2(i + 2) for i, score in enumerate(scores[:k])
        ])

    ndcg_scores = []
    for pred, rel, rel_scores in zip(predicted_indices, relevant_indices, relevant_scores):
        actual_scores = [1 if p in rel else 0 for p in pred]
        ideal_scores = sorted(rel_scores, reverse=True)

        actual_dcg = dcg(actual_scores, k)
        ideal_dcg = dcg(ideal_scores, k)

        ndcg_scores.append(actual_dcg / ideal_dcg if ideal_dcg > 0 else 0)
    return np.mean(ndcg_scores)

# Apply Metrics on the Dataset
k = 10
predicted_movie_indices = []
predicted_movie_ratings = []
relevant_movie_indices = []
relevant_movie_ratings = []

for user_idx in range(num_users):
    user_mask = test_users == user_idx
    # If not a test user skip
    if not np.any(user_mask):
      continue

    # Get predicted ratings for all movies for the user
    predictions = R_pred[user_idx, :].detach().cpu().numpy()
    top_k_predicted_indices = np.argsort(predictions)[-k:][::-1]
    top_k_predicted_ratings = predictions[top_k_predicted_indices][::-1]
    predicted_movie_indices.append(top_k_predicted_indices)
    predicted_movie_ratings.append(top_k_predicted_ratings)

    # movie ids for the user in the test edges
    user_movie_indices = test_movies[user_mask]
    # ratings for movies by user in the test edges
    user_ratings = test_ratings_actual[user_mask]

    # Identify relevant movie indices (ratings >= 3)
    relevant_indices = user_movie_indices[user_ratings >= 3]
    relevant_movie_indices.append(relevant_indices.tolist())
    relevant_movie_ratings.append(user_ratings[user_ratings >= 3].tolist())

# Calculate Precision, Recall, MRR and nDCG
precision = top_k_precision(predicted_movie_indices, predicted_movie_ratings, relevant_movie_indices, relevant_movie_ratings, k)
recall = top_k_recall(predicted_movie_indices, predicted_movie_ratings, relevant_movie_indices, relevant_movie_ratings, k)
mrr = mean_reciprocal_rank(predicted_movie_indices, relevant_movie_indices)
ndcg_score = ndcg(predicted_movie_indices, relevant_movie_indices, relevant_movie_ratings, k)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
print(f"Normalized Discounted Cumulative Gain (nDCG): {ndcg_score:.4f}")

