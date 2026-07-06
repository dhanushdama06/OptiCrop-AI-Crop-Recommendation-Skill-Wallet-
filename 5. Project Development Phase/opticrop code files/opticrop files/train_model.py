import os
import urllib.request
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from models.preprocessing import DataPreprocessor
from models.training import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DATASET_DIR = "dataset"
DATASET_PATH = os.path.join(DATASET_DIR, "crop_recommendation.csv")
DATASET_URL = "https://raw.githubusercontent.com/PAIshanMadusha/crop-recommendation-model/main/dataset/Crop_recommendation.csv"
IMAGES_DIR = os.path.join("static", "images")
MODELS_DIR = "."

def download_dataset():
    """Downloads the dataset if not present locally."""
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        
    if not os.path.exists(DATASET_PATH):
        logger.info(f"Dataset not found locally. Downloading from {DATASET_URL}...")
        try:
            urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
            logger.info("Dataset downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download dataset: {str(e)}")
            raise e
    else:
        logger.info("Dataset found locally.")

def run_eda(df: pd.DataFrame):
    """Performs Exploratory Data Analysis and saves plots to static/images."""
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    logger.info("Running Exploratory Data Analysis (EDA)...")
    
    # 1. Dataset Shape and Description
    logger.info(f"Dataset Shape: {df.shape}")
    desc = df.describe()
    logger.info(f"\nSummary Statistics:\n{desc}")
    
    # 2. Class Distribution Plot
    plt.figure(figsize=(12, 6))
    sns.countplot(y='label', data=df, order=df['label'].value_counts().index, palette='viridis')
    plt.title("Distribution of Crops (Target Label)")
    plt.xlabel("Count")
    plt.ylabel("Crop")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "class_distribution.png"), dpi=100)
    plt.close()
    
    # 3. Correlation Heatmap (excluding label)
    plt.figure(figsize=(10, 8))
    numeric_df = df.drop(columns=['label'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "correlation_heatmap.png"), dpi=100)
    plt.close()
    
    # 4. Feature Histograms (Distributions)
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.histplot(df[feature], ax=axes[i], kde=True, color='forestgreen')
        axes[i].set_title(f"Distribution of {feature}")
        
    # Delete the last empty subplot (since we have 7 features and 8 subplots)
    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "feature_distributions.png"), dpi=100)
    plt.close()

    # 5. Box Plots to visualize outliers
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        sns.boxplot(y=df[feature], ax=axes[i], color='lightgreen')
        axes[i].set_title(f"Boxplot of {feature}")
        
    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "feature_boxplots.png"), dpi=100)
    plt.close()
    
    # 6. Pair plots of first few features to see interactions
    # To keep pairplots neat and fast, we will sample 4 features (N, P, K, pH)
    plt.figure(figsize=(10, 10))
    pairplot = sns.pairplot(df[['N', 'P', 'K', 'ph', 'label']], hue='label', palette='tab20')
    pairplot.fig.suptitle("Pairwise Interactions (N, P, K, pH)", y=1.02)
    pairplot.savefig(os.path.join(IMAGES_DIR, "feature_pairplot.png"), dpi=100)
    plt.close()
    
    logger.info("EDA completed and plots saved to static/images/")

def main():
    # Step 1: Ensure dataset is available
    download_dataset()
    
    # Step 2: Load and Preprocess Data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(DATASET_PATH)
    preprocessor.inspect_data(df)
    
    # Step 3: Run EDA
    run_eda(df)
    
    # Step 4: Full Preprocessing (Clean, Scale, Label Encode)
    X_scaled, y_encoded = preprocessor.preprocess_train(df)
    
    # Step 5: Train-Test Split
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_scaled, y_encoded)
    
    # Step 6: Initialize Trainer and Train Models
    trainer = ModelTrainer(random_state=42)
    
    # Supervised Models Training
    trainer.train_supervised_models(X_train, y_train)
    
    # Unsupervised Model (K-Means) Training
    kmeans_model = trainer.train_unsupervised_model(X_scaled)
    
    # Step 7: Model Evaluation
    eval_df = trainer.evaluate_models(X_test, y_test, X_train, y_train)
    
    # Step 8: Print Comparison Table
    print("\n" + "="*80)
    print("                      MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(eval_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Cross Validation Accuracy']].to_string(index=False))
    print("="*80 + "\n")
    
    # Step 9: Save Comparison Results to a Text file
    results_path = os.path.join(IMAGES_DIR, "model_comparison.txt")
    eval_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Cross Validation Accuracy']].to_csv(results_path, index=False, sep='\t')
    
    # Step 10: Select & Save Best Model and preprocessing artifacts
    best_model, best_name, best_acc = trainer.select_best_model(eval_df)
    
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'model.pkl'))
    joblib.dump(preprocessor.scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(preprocessor.label_encoder, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    joblib.dump(kmeans_model, os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    
    logger.info(f"Saved best model ({best_name}) to model.pkl")
    logger.info(f"Saved Standard Scaler to scaler.pkl")
    logger.info(f"Saved Label Encoder to label_encoder.pkl")
    logger.info(f"Saved K-Means Clustering model to kmeans_model.pkl")
    logger.info("Pipeline executed successfully. All artifacts persisted.")

if __name__ == "__main__":
    main()
