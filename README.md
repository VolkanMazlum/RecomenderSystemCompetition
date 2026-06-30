# Recomender System Competition
Placed in 7 among 80 groups. 

# Hybrid Recommender System

This repository contains a Jupyter Notebook (`Final_ReccomenderSystemHybrid.ipynb`) that implements a robust hybrid recommender system. It combines Graph-based and Machine Learning-based collaborative filtering approaches, along with Content-Based filtering using Item Content Metadata (ICM), to predict and recommend the top 10 items for target users.

## Features

- **Data Preprocessing**: Efficiently loads and converts user-item interactions and item-content metadata into sparse matrices using `scipy.sparse`.
- **Data Splitting**: Utilizes a holdout validation strategy to split the data into training, validation, and test sets.
- **Core Algorithms**:
  - `RP3betaRecommender`: A graph-based collaborative filtering approach.
  - `SLIMElasticNetRecommender`: Sparse Linear Methods with ElasticNet regularization for item recommendation.
- **Hybridization**: Combines the similarity matrices of RP3beta, SLIM ElasticNet, and the Item Content Metadata to form a unified `ItemKNNCustomSimilarityRecommender`.
- **Hyperparameter Tuning**: Includes mechanisms to tune base algorithm parameters (`alpha`, `beta`, `topK`, `l1_ratio`) and systematically searches for the optimal blending weights (alpha) for the hybrid similarity matrix. The notebook also features **Optuna** integration for automated hyperparameter optimization.
- **Evaluation**: Computes comprehensive evaluation metrics (focusing on MAP - Mean Average Precision) using a custom `EvaluatorHoldout`.

## Requirements

The notebook dynamically clones the required `RecSys_Course_AT_PoliMi` framework repository and installs necessary dependencies. Key packages include:

- Python 
- pandas
- numpy
- scipy
- scikit-learn
- optuna

## Usage

1. **Data Placement**: Ensure that the datasets (`data_train.csv`, `data_ICM_metadata.csv`, `data_target_users_test.csv`) are placed in the directory paths referenced in the notebook. You might need to update the file paths (`folder_path`) according to your local setup.
2. **Execution**: Run the Jupyter Notebook cells sequentially. The initial cells will automatically clone the required Polimi RecSys repository and install missing libraries.
3. **Output**: Once the execution and evaluation are complete, the notebook will generate a submission file (e.g., `sample_submission.csv` or `sample_submission_optuna.csv`) containing the top 10 recommendations for each target user.
