import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.target_col = 'label'

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Loads the dataset from a CSV file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found at {filepath}")
        logger.info(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df

    def inspect_data(self, df: pd.DataFrame):
        """Displays basic dataset information and statistics."""
        logger.info("--- Dataset Summary Statistics ---")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        logger.info(f"Duplicate rows count: {df.duplicated().sum()}")
        logger.info(f"Unique target labels count: {df[self.target_col].nunique()}")
        logger.info(f"Target labels: {df[self.target_col].unique()}")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the data by handling null values and duplicates."""
        df_cleaned = df.copy()
        
        # Check and handle duplicates
        duplicates = df_cleaned.duplicated().sum()
        if duplicates > 0:
            logger.info(f"Removing {duplicates} duplicate rows...")
            df_cleaned = df_cleaned.drop_duplicates()
        
        # Handle missing values if any
        null_counts = df_cleaned.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning("Missing values detected. Imputing numerical columns with median...")
            for col in self.feature_cols:
                if df_cleaned[col].isnull().sum() > 0:
                    median_val = df_cleaned[col].median()
                    df_cleaned[col].fillna(median_val, inplace=True)
        else:
            logger.info("No missing values found in the dataset.")
            
        return df_cleaned

    def detect_and_handle_outliers(self, df: pd.DataFrame, method='iqr', action='log') -> pd.DataFrame:
        """
        Detects outliers in features.
        action='log': Log outlier statistics.
        action='cap': Cap outliers to 1.5 * IQR bounds.
        Note: In agricultural datasets, high nutrients or rainfall can represent valid crop requirements.
        Therefore, we log them but do not aggressively discard them by default, to avoid removing rare crops.
        """
        df_out = df.copy()
        logger.info("Detecting outliers in features...")
        
        for col in self.feature_cols:
            q1 = df_out[col].quantile(0.25)
            q3 = df_out[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df_out[(df_out[col] < lower_bound) | (df_out[col] > upper_bound)]
            logger.info(f"Feature '{col}': found {len(outliers)} outliers out of {len(df_out)} rows.")
            
            if action == 'cap' and len(outliers) > 0:
                logger.info(f"Capping outliers in '{col}' to range [{lower_bound:.2f}, {upper_bound:.2f}]")
                df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)
                
        return df_out

    def preprocess_train(self, df: pd.DataFrame):
        """
        Preprocesses raw dataframe for training:
        1. Clean duplicates and missing values
        2. Detect/Log outliers
        3. Split features (X) and target (y)
        4. Fit and apply label encoder to y
        5. Fit and apply StandardScaler to X
        Returns: X_scaled, y_encoded
        """
        cleaned_df = self.clean_data(df)
        processed_df = self.detect_and_handle_outliers(cleaned_df, action='log')
        
        X = processed_df[self.feature_cols]
        y = processed_df[self.target_col]
        
        # Fit Label Encoder
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Fit Scaler
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_cols)
        
        return X_scaled_df, y_encoded

    def preprocess_test(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies pre-fit feature scaler to new or test data."""
        X_scaled = self.scaler.transform(X[self.feature_cols])
        return pd.DataFrame(X_scaled, columns=self.feature_cols)

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Splits the preprocessed dataset into train and test sets."""
        logger.info(f"Splitting data with test_size={test_size}...")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
