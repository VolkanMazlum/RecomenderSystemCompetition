import os
import csv
import scipy.sparse as sp
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import itertools
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_train = pd.read_csv("C:\\Users\\VOLKAN MAZLUM\\Desktop\\Proje\\data_train.csv")
icm_metadata = pd.read_csv("C:\\Users\\VOLKAN MAZLUM\\Desktop\\Proje\\data_ICM_metadata.csv")
target_users = pd.read_csv("C:\\Users\\VOLKAN MAZLUM\\Desktop\\Proje\\data_target_users_test.csv")


# Process item-content metadata (ICM)
icm_column_names = ['item_id', 'feature_id', 'value']
icm_df = icm_metadata.rename(columns={icm_metadata.columns[0]: 'item_id', 
                                       icm_metadata.columns[1]: 'feature_id', 
                                       icm_metadata.columns[2]: 'value'})

ICM_all = sp.coo_matrix(
    (icm_df['value'].values, 
     (icm_df['item_id'].values, icm_df['feature_id'].values))

     
)

# Exploratory Data Analysis (EDA)
def perform_eda(data_train, icm_metadata, target_users):
    print("Performing EDA...\n")
    
    # Data overview
    print("### Data Train Overview ###")
    print(data_train.info())
    print(data_train.describe())
    print(f"Number of unique users: {data_train['user_id'].nunique()}")
    print(f"Number of unique items: {data_train['item_id'].nunique()}")
    print(f"Total interactions: {len(data_train)}\n")
    
    # Interaction density
    n_users = data_train['user_id'].nunique()
    n_items = data_train['item_id'].nunique()
    interaction_density = len(data_train) / (n_users * n_items) * 100
    print(f"Interaction Density: {interaction_density:.4f}%\n")
    
    # Distribution of interactions per user
    print("### Interactions per User ###")
    interactions_per_user = data_train.groupby('user_id').size()
    print(interactions_per_user.describe())
    interactions_per_user.hist(bins=50, alpha=0.7, color='blue')
    plt.title("Interactions per User")
    plt.xlabel("Number of Interactions")
    plt.ylabel("Frequency")
    plt.show()
    
    # Distribution of interactions per item
    print("### Interactions per Item ###")
    interactions_per_item = data_train.groupby('item_id').size()
    print(interactions_per_item.describe())
    interactions_per_item.hist(bins=50, alpha=0.7, color='green')
    plt.title("Interactions per Item")
    plt.xlabel("Number of Interactions")
    plt.ylabel("Frequency")
    plt.show()
    
    # ICM Metadata Analysis
    print("### ICM Metadata Overview ###")
    print(icm_metadata.info())
    print(icm_metadata.describe())
    print(f"Number of unique items in ICM: {icm_metadata['item_id'].nunique()}")
    print(f"Number of unique features: {icm_metadata['feature_id'].nunique()}")
    print(f"Total item-feature interactions: {len(icm_metadata)}\n")
    
    # Target Users Analysis
    print("### Target Users Overview ###")
    print(target_users.info())
    print(f"Number of target users: {target_users['user_id'].nunique()}\n")
    
    print("EDA Completed.\n")

# Call the EDA function
perform_eda(data_train, icm_metadata, target_users)


# Function for Exploratory Data Analysis (EDA)
def perform_eda(data):
    print("Dataset Info:")
    print(data.info())

    print("\nDataset Description:")
    print(data.describe())

    # Check for missing values
    missing_values = data.isnull().sum()
    print("\nMissing Values:")
    print(missing_values)

    # Distribution of interactions per user
    user_interactions = data['user_id'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.hist(user_interactions, bins=50, color='blue', alpha=0.7)
    plt.title("Distribution of Interactions per User")
    plt.xlabel("Number of Interactions")
    plt.ylabel("Frequency")
    plt.show()

    # Distribution of interactions per item
    item_interactions = data['item_id'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.hist(item_interactions, bins=50, color='green', alpha=0.7)
    plt.title("Distribution of Interactions per Item")
    plt.xlabel("Number of Interactions")
    plt.ylabel("Frequency")
    plt.show()

    # Correlation heatmap (if applicable)
    if 'data' in data.columns:
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()

# Perform EDA
perform_eda(data_train)

# Create sparse matrix
n_users = data_train['user_id'].nunique()
n_items = data_train['item_id'].nunique()

URM_all = csr_matrix((data_train['data'], 
                      (data_train['user_id'], data_train['item_id'])),
                     shape=(n_users, n_items))

# Split data
def split_data(URM_all, train_percentage=0.8):
    URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage)
    return URM_train, URM_validation, URM_test

# Grid Search for Hyperparameter Tuning
def grid_search(URM_train, URM_validation, param_grid):
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    
    best_score = -np.inf
    best_params = None

    for params in itertools.product(*param_grid.values()):
        current_params = dict(zip(param_grid.keys(), params))

        # Train RP3beta
        rp3_recommender = RP3betaRecommender(URM_train)
        rp3_recommender.fit(alpha=current_params['alpha_rp3'], beta=current_params['beta_rp3'], topK=current_params['topK_rp3'])

        # Train SLIMElasticNet
        slim_recommender = SLIMElasticNetRecommender(URM_train)
        slim_recommender.fit(alpha=current_params['alpha_slim'], l1_ratio=current_params['l1_ratio_slim'], topK=current_params['topK_slim'])

        # Combine recommendations using hybrid alpha
        hybrid_similarity = ((1 - current_params['hybrid_alpha']) * rp3_recommender.W_sparse 
                             + current_params['hybrid_alpha'] * slim_recommender.W_sparse)

        hybrid_recommender = ItemKNNCustomSimilarityRecommender(URM_train)
        hybrid_recommender.fit(hybrid_similarity)

        # Evaluate on validation set
        result_dict, _ = evaluator_validation.evaluateRecommender(hybrid_recommender)
        map_score = result_dict["MAP"].values[0]

        if map_score > best_score:
            best_score = map_score
            best_params = current_params

    return best_params, best_score

# Define parameter grid
param_grid = {
    "alpha_rp3": [0.05, 0.1, 0.5],
    "beta_rp3": [0.01, 0.1, 0.5],
    "topK_rp3": [10, 50, 75],
    "alpha_slim": [1e-5, 1e-4, 1e-3],
    "l1_ratio_slim": [0.1, 0.5, 0.9],
    "topK_slim": [1500, 2000, 2500],
    "hybrid_alpha": [0.1, 0.5, 0.9]
}

# Main workflow
def main(URM_all):
    global URM_train, URM_validation
    
    # Split data
    URM_train, URM_validation, URM_test = split_data(URM_all)
    
    # Perform Grid Search
    best_params, best_score = grid_search(URM_train, URM_validation, param_grid)
    print("Best hyperparameters:", best_params)
    print("Best MAP:", best_score)
    
    # Retrain on full data with best parameters
    rp3_recommender = RP3betaRecommender(URM_all)
    rp3_recommender.fit(alpha=best_params["alpha_rp3"], beta=best_params["beta_rp3"], topK=best_params["topK_rp3"])
    
    slim_recommender = SLIMElasticNetRecommender(URM_all)
    slim_recommender.fit(alpha=best_params["alpha_slim"], l1_ratio=best_params["l1_ratio_slim"], topK=best_params["topK_slim"])
    
    hybrid_similarity = ((1 - best_params["hybrid_alpha"]) * rp3_recommender.W_sparse 
                         + best_params["hybrid_alpha"] * slim_recommender.W_sparse)
                         
    hybrid_recommender = ItemKNNCustomSimilarityRecommender(URM_all)
    hybrid_recommender.fit(hybrid_similarity)
    
    return hybrid_recommender

# Run the pipeline
hybrid_recommender = main(URM_all)

# Save results
output_folder = "C:\\Users\\VOLKAN MAZLUM\\Desktop\\Proje\\"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "sample_submission_grid_search.csv")

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'item_list'])
    for user_id in target_users['user_id']:
        recommendations = hybrid_recommender.recommend(user_id, cutoff=10)
        writer.writerow([user_id, " ".join(map(str, recommendations))])

print(f"Results saved to {output_file}")
