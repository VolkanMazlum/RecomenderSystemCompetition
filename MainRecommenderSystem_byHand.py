
# Main Recommender System Script with EDA

# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Exploratory Data Analysis (EDA) Section
# Load your dataset here (Example placeholder)
# df = pd.read_csv('your_dataset.csv')  # Replace with actual dataset

# EDA Steps
# print(df.info())  # Dataset overview
# print(df.describe())  # Summary statistics

# Visualize Missing Values
# missing_values = df.isnull().sum()
# print(missing_values)

# Correlation Matrix
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()



import pandas as pd
import scipy.sparse as sp
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender

# Load datasets
data_train = pd.read_csv("C:\\Users\\VOLKAN MAZLUM\\Desktop\\Proje\\data_train.csv")
icm_metadata = pd.read_csv("C:\\Users\\VOLKAN MAZLUM\\Desktop\\Proje\\data_ICM_metadata.csv")
target_users = pd.read_csv("C:\\Users\\VOLKAN MAZLUM\\Desktop\\Proje\\data_target_users_test.csv")

# Process user-item interaction data
column_names = ['user_id', 'item_id', 'interaction']
dataframe = data_train.rename(columns={data_train.columns[0]: 'user_id', 
                                       data_train.columns[1]: 'item_id', 
                                       data_train.columns[2]: 'interaction'})

URM_all = sp.coo_matrix(
    (dataframe['interaction'].values, 
     (dataframe['user_id'].values, dataframe['item_id'].values))
)

# Process item-content metadata (ICM)
icm_column_names = ['item_id', 'feature_id', 'value']
icm_df = icm_metadata.rename(columns={icm_metadata.columns[0]: 'item_id', 
                                       icm_metadata.columns[1]: 'feature_id', 
                                       icm_metadata.columns[2]: 'value'})

ICM_all = sp.coo_matrix(
    (icm_df['value'].values, 
     (icm_df['item_id'].values, icm_df['feature_id'].values))
)




# Split data into train, validation, and test
URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage=0.8)

# Set up evaluators
evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])



# Define RP3beta recommender
rp3_params = {"alpha": 0.336, "beta": 0.146, "topK": 32}  # Example parameters
recommender_rp3 = RP3betaRecommender(URM_train)
recommender_rp3.fit(**rp3_params)

# Define SLIM ElasticNet recommender
slim_params = {"topK": 2305, "l1_ratio": 0.16, "alpha": 0.00069}  # Example parameters
recommender_slim = SLIMElasticNetRecommender(URM_train)
recommender_slim.fit(**slim_params)

# Evaluate individual recommenders
print("RP3beta validation results:")
results_rp3, _ = evaluator_validation.evaluateRecommender(recommender_rp3)
print(results_rp3)

print("SLIM validation results:")
results_slim, _ = evaluator_validation.evaluateRecommender(recommender_slim)
print(results_slim)



# Create a hybrid recommender using weighted similarity matrices
alpha_hybrid = 0.56  # Weight for combining RP3 and SLIM similarities
hybrid_similarity = (1 - alpha_hybrid) * recommender_rp3.W_sparse + alpha_hybrid * recommender_slim.W_sparse

recommender_hybrid = ItemKNNCustomSimilarityRecommender(URM_train)
recommender_hybrid.fit(hybrid_similarity)



# Evaluate the hybrid recommender
print("Hybrid recommender validation results:")
results_hybrid, _ = evaluator_validation.evaluateRecommender(recommender_hybrid)
print(results_hybrid)

folder_path = "C:\\Users\\VOLKAN MAZLUM\\Desktop\\Proje\\"
import os
import csv
os.makedirs(folder_path, exist_ok=True)

output_file = os.path.join(folder_path, "sample_submission.csv")
# Generate recommendations for target users

with open(output_file, 'w') as f:
    f.write("user_id,item_list\n")
    for user_id in target_users['user_id']:
        recommended_items = recommender_hybrid.recommend(user_id, cutoff=10)
        f.write(f"{user_id},{' '.join(map(str, recommended_items))}\n")

print(f"Recommendations saved to {output_file}")

import os
import csv
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


import pandas as pd
import scipy.sparse as sp
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender

# Veri yükleme
data_train = pd.read_csv("C:\\Users\\VOLKAN MAZLUM\\Desktop\\Proje\\data_train.csv")
icm_metadata = pd.read_csv("C:\\Users\\VOLKAN MAZLUM\\Desktop\\Proje\\data_ICM_metadata.csv")
target_users = pd.read_csv("C:\\Users\\VOLKAN MAZLUM\\Desktop\\Proje\\data_target_users_test.csv")

# Sparse matrisi oluşturma
n_users = data_train['user_id'].nunique()
n_items = data_train['item_id'].nunique()

URM_all = csr_matrix((data_train['data'], 
                      (data_train['user_id'], data_train['item_id'])),
                     shape=(n_users, n_items))
n_features = icm_metadata['feature_id'].nunique()

# Create the sparse matrix for ICM (Item-Feature interactions)
ICM_all = csr_matrix((icm_metadata['data'], 
                      (icm_metadata['item_id'], icm_metadata['feature_id'])),
                     shape=(n_items, n_features))

# Split data fonksiyonu
def split_data(URM_all, train_percentage=0.8):
    URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage)
    return URM_train, URM_validation, URM_test

# En iyi alpha'yı bulma fonksiyonu
def find_best_alpha(recommender1, recommender2, evaluator, alpha_range, URM_train):
    best_alpha, best_map = None, 0.0
    for alpha in alpha_range:
        hybrid_similarity = (1 - alpha) * recommender1.W_sparse + alpha * recommender2.W_sparse
        hybrid_recommender = ItemKNNCustomSimilarityRecommender(URM_train)
        hybrid_recommender.fit(hybrid_similarity)
        
        results_df, _ = evaluator.evaluateRecommender(hybrid_recommender)
        map_value = results_df["MAP"].values[0]
        
        if map_value > best_map:
            best_alpha, best_map = alpha, map_value
    
    return best_alpha, best_map

# Hiperparametre arama fonksiyonu
def tune_recommender_params(recommender_class, params_list, evaluator, URM_train):
    best_index, max_map = None, 0.0
    for i, params in enumerate(params_list):
        recommender = recommender_class(URM_train)
        recommender.fit(**params)
        
        results_df, _ = evaluator.evaluateRecommender(recommender)
        map_value = results_df["MAP"].item()
        
        if map_value > max_map:
            best_index, max_map = i, map_value
    
    return best_index

# Ana iş akışı
def main(URM_all, params_rp3, params_slim, alpha_range):
    # Veri bölme
    URM_train, URM_validation, URM_test = split_data(URM_all)
    
    # Değerlendirme objeleri
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
    
    # RP3 için en iyi parametre seçimi
    rp3_index = tune_recommender_params(RP3betaRecommender, params_rp3, evaluator_validation, URM_train)
    best_rp3_params = params_rp3[rp3_index]
    rp3_recommender = RP3betaRecommender(URM_train)
    rp3_recommender.fit(**best_rp3_params)
    
    # SLIM için en iyi parametre seçimi
    slim_index = tune_recommender_params(SLIMElasticNetRecommender, params_slim, evaluator_validation, URM_train)
    best_slim_params = params_slim[slim_index]
    slim_recommender = SLIMElasticNetRecommender(URM_train)
    slim_recommender.fit(**best_slim_params)
    
    # En iyi alpha değerini bulma
    best_alpha, best_map = find_best_alpha(rp3_recommender, slim_recommender, evaluator_validation, alpha_range,URM_train)
    print(f"Best alpha: {best_alpha}, Best MAP: {best_map}")
    
    # Tüm veriyle yeniden eğitim ve tahmin
    rp3_recommender = RP3betaRecommender(URM_all)
    rp3_recommender.fit(**best_rp3_params)
    slim_recommender = SLIMElasticNetRecommender(URM_all)
    slim_recommender.fit(**best_slim_params)
    
    hybrid_similarity = (1 - best_alpha) * rp3_recommender.W_sparse + best_alpha * slim_recommender.W_sparse
    hybrid_recommender = ItemKNNCustomSimilarityRecommender(URM_all)
    hybrid_recommender.fit(hybrid_similarity)
    
    return hybrid_recommender



# Örnek parametreler
params_rp3 = [{"alpha": 0.33, "beta": 0.14, "topK": 32}, {"alpha": 0.32, "beta": 0.12, "topK": 30}, {"alpha": 0.39, "beta": 0.14, "topK": 28}, {"alpha": 0.32, "beta": 0.14, "topK": 36}, {"alpha": 0.35, "beta": 0.16, "topK": 35}]
params_slim = [{"topK": 2305, "l1_ratio": 0.15, "alpha": 0.0006}, {"topK": 2327, "l1_ratio": 0.16, "alpha": 0.0007}, {"topK": 1827, "l1_ratio": 0.19, "alpha": 0.0005}, {"topK": 2007, "l1_ratio": 0.11, "alpha": 0.0009}, {"topK": 3327, "l1_ratio": 0.17, "alpha": 0.0012}]
alpha_range = np.arange(0.1, 0.68, 0.01)

# Çalıştırma
hybrid_recommender = main(URM_all, params_rp3, params_slim, alpha_range)

# Sonuçları kaydetme
output_folder = "C:\\Users\\VOLKAN MAZLUM\\Desktop\\Proje\\"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "sample_submission11.csv")

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'item_list'])
    for user_id in target_users['user_id']:
        recommendations = hybrid_recommender.recommend(user_id, cutoff=10)
        writer.writerow([user_id, " ".join(map(str, recommendations))])

print(f"Results saved to {output_file}")



# Örnek parametreler
params_rp3 = [{"alpha": 0.38, "beta": 0.25, "topK": 87},{"alpha": 0.32, "beta": 0.11, "topK": 30},{"alpha": 0.35, "beta": 0.24, "topK": 28},{"alpha": 0.54, "beta": 0.11, "topK": 49},{"alpha": 0.35, "beta": 0.21, "topK": 45},{"alpha": 0.33, "beta": 0.14, "topK": 32}, {"alpha": 0.32, "beta": 0.12, "topK": 30}, {"alpha": 0.39, "beta": 0.14, "topK": 28}, {"alpha": 0.32, "beta": 0.14, "topK": 36}, {"alpha": 0.35, "beta": 0.16, "topK": 35}, {"alpha": 0.55, "beta": 0.10, "topK": 26}]
params_slim = [{"topK": 2305, "l1_ratio": 0.15, "alpha": 0.0006}, {"topK": 2327, "l1_ratio": 0.16, "alpha": 0.0007}, {"topK": 1827, "l1_ratio": 0.19, "alpha": 0.0005}, {"topK": 2007, "l1_ratio": 0.11, "alpha": 0.0009}, {"topK": 3327, "l1_ratio": 0.17, "alpha": 0.0012}]
alpha_range = np.arange(0.1, 0.68, 0.01)

# Çalıştırma
hybrid_recommender = main(URM_all, params_rp3, params_slim, alpha_range)

# Sonuçları kaydetme
output_folder = "C:\\Users\\VOLKAN MAZLUM\\Desktop\\Proje\\"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "sample_submission12.csv")

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'item_list'])
    for user_id in target_users['user_id']:
        recommendations = hybrid_recommender.recommend(user_id, cutoff=10)
        writer.writerow([user_id, " ".join(map(str, recommendations))])

print(f"Results saved to {output_file}")



# Örnek parametreler
params_rp3 = [{"alpha": 0.38, "beta": 0.25, "topK": 87},{"alpha": 0.32, "beta": 0.11, "topK": 30},{"alpha": 0.35, "beta": 0.24, "topK": 28},{"alpha": 0.54, "beta": 0.11, "topK": 49},{"alpha": 0.35, "beta": 0.21, "topK": 45},{"alpha": 0.33, "beta": 0.14, "topK": 32}, {"alpha": 0.32, "beta": 0.12, "topK": 30}, {"alpha": 0.39, "beta": 0.14, "topK": 28}, {"alpha": 0.32, "beta": 0.14, "topK": 36}, {"alpha": 0.35, "beta": 0.16, "topK": 35}, {"alpha": 0.55, "beta": 0.10, "topK": 26}]
params_slim = [{"topK": 2455, "l1_ratio": 0.20, "alpha": 0.0010},{"topK": 2305, "l1_ratio": 0.19, "alpha": 0.0018},{"topK": 2305, "l1_ratio": 0.15, "alpha": 0.0006}, {"topK": 2327, "l1_ratio": 0.16, "alpha": 0.0007}, {"topK": 1827, "l1_ratio": 0.19, "alpha": 0.0005}, {"topK": 2007, "l1_ratio": 0.11, "alpha": 0.0009}, {"topK": 3327, "l1_ratio": 0.17, "alpha": 0.0012}]
alpha_range = np.arange(0.1, 0.68, 0.01)

# Çalıştırma
hybrid_recommender = main(URM_all, params_rp3, params_slim, alpha_range)

# Sonuçları kaydetme
output_folder = "C:\\Users\\VOLKAN MAZLUM\\Desktop\\Proje\\"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "sample_submission10.csv")

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'item_list'])
    for user_id in target_users['user_id']:
        recommendations = hybrid_recommender.recommend(user_id, cutoff=10)
        writer.writerow([user_id, " ".join(map(str, recommendations))])

print(f"Results saved to {output_file}")



# Örnek parametreler
params_rp3 = [{"alpha": 0.3365280890390189, "beta": 0.14559541984971103, "topK": 32},
    {"alpha": 0.33543320721534375, "beta": 0.13643832833854405, "topK": 31},
    {"alpha": 0.3227577390094315, "beta": 0.12631978308173344, "topK": 32},
    {"alpha": 0.2617995480887691, "beta": 0.22418804987168886, "topK": 30},
    {'alpha': 0.28585975670634217, 'beta': 0.13388489746818844, 'topK': 33},
    {'alpha': 0.3361086178381283, 'beta': 0.13949133462799973, 'topK': 29}, 
    {"alpha": 0.32, "beta": 0.12, "topK": 30}, 
    {"alpha": 0.39, "beta": 0.14, "topK": 28}, 
    {"alpha": 0.32, "beta": 0.14, "topK": 36}, 
    {"alpha": 0.35, "beta": 0.16, "topK": 35}, 
    {"alpha": 0.55, "beta": 0.10, "topK": 26}]
params_slim = [{'topK': 2305, 'l1_ratio': 0.15984659917724292, 'alpha': 0.0006895792558081994},
    {'topK': 2339, 'l1_ratio': 0.15486907556362542, 'alpha': 0.0006851706335261893},
    {"topK": 2327, "l1_ratio": 0.15346747937279875, "alpha": 0.000677913689441996},
    {'topK': 2427, 'l1_ratio': 0.14931044947790595, 'alpha': 0.0007442377587336158},
    {'topK': 2310, 'l1_ratio': 0.1519150334556062, 'alpha': 0.0006862030334431442}
    ]
alpha_range = np.arange(0.25, 0.70, 0.01)

# Çalıştırma
hybrid_recommender = main(URM_all, params_rp3, params_slim, alpha_range)

# Sonuçları kaydetme
output_folder = "C:\\Users\\VOLKAN MAZLUM\\Desktop\\Proje\\"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "sample_submission14.csv")

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'item_list'])
    for user_id in target_users['user_id']:
        recommendations = hybrid_recommender.recommend(user_id, cutoff=10)
        writer.writerow([user_id, " ".join(map(str, recommendations))])

print(f"Results saved to {output_file}")
