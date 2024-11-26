import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from pyspark.ml.recommendation import ALS as SparkALS
from pyspark.sql import SparkSession
from torch_geometric.datasets import MovieLens100K
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

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

num_users = user_features.shape[0]
num_movies = movie_features.shape[0]

# Convert train, val, and test nodes to masks
train_mask = torch.isin(torch.tensor(data_edge_index[0].numpy()), torch.tensor(train_nodes))
val_mask = torch.isin(torch.tensor(data_edge_index[0].numpy()), torch.tensor(val_nodes))
test_mask = torch.isin(torch.tensor(data_edge_index[0].numpy()), torch.tensor(test_nodes))

# Prepare user-item interaction data
user_indices = data_edge_index[0].numpy()
movie_indices = data_edge_index[1].numpy()
ratings = data_edge_label.numpy()

# Create sparse matrices for train, validation, and test sets
train_matrix = coo_matrix(
    (ratings, (user_indices, movie_indices)),
    shape=(num_users, num_movies)
)

test_matrix = coo_matrix(
    (ratings[test_mask], (user_indices[test_mask], movie_indices[test_mask])),
    shape=(num_users, num_movies)
)

# Initialize Spark session
spark = SparkSession.builder.master("local").appName("ALS Example").getOrCreate()

# Prepare the ALS training and test data from the train, validation, and test splits
train_data = np.vstack([user_indices, movie_indices, ratings]).T
val_data = np.vstack([user_indices[val_mask], movie_indices[val_mask], ratings[val_mask]]).T
test_data = np.vstack([user_indices[test_mask], movie_indices[test_mask], ratings[test_mask]]).T

# Convert training data to Spark DataFrame
train_df = spark.createDataFrame(
    [(int(row[0]), int(row[1]), float(row[2])) for row in train_data],
    schema=["userId", "movieId", "rating"]
)

val_df = spark.createDataFrame(
    [(int(row[0]), int(row[1]), float(row[2])) for row in val_data],
    schema=["userId", "movieId", "rating"]
)

test_df = spark.createDataFrame(
    [(int(row[0]), int(row[1]), float(row[2])) for row in test_data],
    schema=["userId", "movieId", "rating"]
)

# Configure ALS model
als = SparkALS(
    maxIter=10,
    regParam=0.1,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop"
)

# Train ALS model
model = als.fit(train_df)

# Evaluate model using RMSE
evaluator = RegressionEvaluator(
    metricName="rmse", labelCol="rating", predictionCol="prediction"
)

# Get predicted ratings for the validation set
val_predictions = model.transform(val_df)
# Round the values and clip them between 0 and 5 in the 'prediction' column
val_predictions = val_predictions.withColumn(
    "prediction",
    F.when(F.round(val_predictions["prediction"]) < 0, 0)
    .when(F.round(val_predictions["prediction"]) > 5, 5)
    .otherwise(F.round(val_predictions["prediction"]))
)

val_rmse = evaluator.evaluate(val_predictions)
print(f"Validation RMSE: {val_rmse:.4f}")

# Get predicted ratings for the test set
test_predictions = model.transform(test_df)
test_predictions = test_predictions.withColumn(
    "prediction",
    F.when(F.round(test_predictions["prediction"]) < 0, 0)
    .when(F.round(test_predictions["prediction"]) > 5, 5)
    .otherwise(F.round(test_predictions["prediction"]))
)
test_rmse = evaluator.evaluate(test_predictions)
print(f"Test RMSE: {test_rmse:.4f}")

# Extract user and item factors from the model
user_factors_df = model.userFactors
item_factors_df = model.itemFactors

# Convert the factors into NumPy arrays
user_factors = user_factors_df.collect()  # Collect to a list
item_factors = item_factors_df.collect()  # Collect to a list

# Create matrices of features
user_matrix = np.vstack([np.array(u["features"]) for u in user_factors])
item_matrix = np.vstack([np.array(i["features"]) for i in item_factors])

# Compute predicted ratings
R_pred = np.dot(user_matrix, item_matrix.T)
R_pred = np.round(R_pred).clip(0, 5).astype(int)

def extract_metrics_data(test_predictions, test_matrix, k):
    test_predictions = test_predictions.orderBy("userId", "prediction", ascending=[True, False])
    user_predictions = {}
    user_relevants = {}

    # Extract predicted items for each user
    for row in test_predictions.collect():
        user = row["userId"]
        movie = row["movieId"]
        pred_rating = row["prediction"]

        if user not in user_predictions:
            user_predictions[user] = []
        if len(user_predictions[user]) < k:
            user_predictions[user].append((movie, pred_rating))

    # Extract relevant items from the test matrix for each user
    test_matrix_row = test_matrix.row
    test_matrix_col = test_matrix.col
    test_matrix_data = test_matrix.data

    for user in range(test_matrix.shape[0]):
        # Find the indices of relevant movies for the current user
        relevant_movies = test_matrix_col[test_matrix_row == user]
        relevant_ratings = test_matrix_data[test_matrix_row == user]
        user_relevants[user] = (relevant_movies, relevant_ratings)

    # Prepare the format required for the metrics functions
    predicted_indices = [np.array([x[0] for x in user_predictions[u]]) for u in user_predictions]
    predicted_ratings = [np.array([x[1] for x in user_predictions[u]]) for u in user_predictions]
    relevant_indices = [user_relevants[u][0] for u in user_relevants]
    relevant_ratings = [user_relevants[u][1] for u in user_relevants]

    return predicted_indices, predicted_ratings, relevant_indices, relevant_ratings

# Metrics Implementation
# Top-k Precision and Recall functions (Exact match of both movie index and rating)
def top_k_precision(predicted_indices, predicted_ratings, relevant_indices, relevant_ratings, k):
  precisions = []
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
        return np.sum([score / np.log2(i + 2) for i, score in enumerate(scores[:k])])

    ndcg_scores = []
    for pred, rel, rel_scores in zip(predicted_indices, relevant_indices, relevant_scores):
        actual_scores = [1 if p in rel else 0 for p in pred]
        ideal_scores = sorted(rel_scores, reverse=True)

        actual_dcg = dcg(actual_scores, k)
        ideal_dcg = dcg(ideal_scores, k)

        ndcg_scores.append(actual_dcg / ideal_dcg if ideal_dcg > 0 else 0)
    return np.mean(ndcg_scores)

k = 10
predicted_indices, predicted_ratings, relevant_indices, relevant_ratings = extract_metrics_data(
    test_predictions, test_matrix, k
)
precision = top_k_precision(predicted_indices, predicted_ratings, relevant_indices, relevant_ratings, k)
recall = top_k_recall(predicted_indices, predicted_ratings, relevant_indices, relevant_ratings, k)
mrr = mean_reciprocal_rank(predicted_indices, relevant_indices)
ndcg_score = ndcg(predicted_indices, relevant_indices, relevant_ratings, k)

print(f"Top-{k} Precision: {precision:.4f}")
print(f"Top-{k} Recall: {recall:.4f}")
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
print(f"Normalized Discounted Cumulative Gain (NDCG): {ndcg_score:.4f}")
