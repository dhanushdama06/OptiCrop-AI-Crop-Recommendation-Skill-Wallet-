import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        }
        self.kmeans = KMeans(n_clusters=22, random_state=self.random_state, n_init=10) # 22 crops in standard dataset
        self.trained_models = {}

    def train_supervised_models(self, X_train, y_train):
        """Trains all supervised models."""
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            logger.info(f"{name} trained successfully.")
            
    def train_unsupervised_model(self, X):
        """Trains the K-Means clustering model."""
        logger.info("Training K-Means Clustering...")
        self.kmeans.fit(X)
        logger.info("K-Means Clustering trained successfully.")
        return self.kmeans

    def evaluate_models(self, X_test, y_test, X_train=None, y_train=None) -> pd.DataFrame:
        """
        Evaluates all trained supervised models.
        Computes Accuracy, Precision, Recall, F1 Score, and Cross-Validation score.
        """
        evaluation_results = []
        
        for name, model in self.trained_models.items():
            logger.info(f"Evaluating {name}...")
            y_pred = model.predict(X_test)
            
            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Compute Cross-Validation score (using train set if provided, else test set)
            X_cv = X_train if X_train is not None else X_test
            y_cv = y_train if y_train is not None else y_test
            cv_scores = cross_val_score(model, X_cv, y_cv, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            
            # Generate classification report and confusion matrix
            class_report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            evaluation_results.append({
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Cross Validation Accuracy": cv_mean,
                "Classification Report": class_report,
                "Confusion Matrix": conf_matrix
            })
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, CV Accuracy: {cv_mean:.4f}")
            
        return pd.DataFrame(evaluation_results)

    def select_best_model(self, eval_df: pd.DataFrame):
        """Selects the model with the highest accuracy."""
        best_row = eval_df.loc[eval_df['Accuracy'].idxmax()]
        best_model_name = best_row['Model']
        best_accuracy = best_row['Accuracy']
        
        logger.info(f"Best model selected: {best_model_name} with Accuracy of {best_accuracy:.4f}")
        return self.trained_models[best_model_name], best_model_name, best_accuracy
