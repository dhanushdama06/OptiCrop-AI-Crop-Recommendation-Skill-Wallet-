import os
import joblib
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PredictionError(Exception):
    """Custom exception class for prediction errors."""
    pass

class CropPredictor:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            # Default to the workspace root or parent of models/
            model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        self.model_path = os.path.join(model_dir, 'model.pkl')
        self.scaler_path = os.path.join(model_dir, 'scaler.pkl')
        self.label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

    def load_artifacts(self):
        """Loads model, scaler, and label encoder from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train the model first.")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {self.scaler_path}. Please train the model first.")
        if not os.path.exists(self.label_encoder_path):
            raise FileNotFoundError(f"Label encoder file not found at {self.label_encoder_path}. Please train the model first.")

        try:
            logger.info("Loading saved model artifacts...")
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.label_encoder = joblib.load(self.label_encoder_path)
            logger.info("Model artifacts loaded successfully.")
        except Exception as e:
            raise PredictionError(f"Error loading model artifacts: {str(e)}")

    def validate_inputs(self, data: dict) -> dict:
        """
        Validates input values.
        Raises PredictionError if validation fails.
        """
        validated_data = {}
        required_fields = self.feature_cols
        
        # 1. Check for missing keys
        for field in required_fields:
            if field not in data:
                raise PredictionError(f"Missing input field: {field}")
            
            val = data[field]
            
            # 2. Check for empty values
            if val is None or str(val).strip() == "":
                raise PredictionError(f"Field '{field}' cannot be empty.")
            
            # 3. Convert to float and validate type
            try:
                float_val = float(val)
            except ValueError:
                raise PredictionError(f"Field '{field}' must be a valid number. Got: '{val}'")
            
            # 4. Check for negative values
            if float_val < 0:
                raise PredictionError(f"Field '{field}' cannot be negative. Got: {float_val}")
                
            validated_data[field] = float_val

        # 5. Domain-specific validation (pH range)
        ph_val = validated_data['ph']
        if ph_val < 0.0 or ph_val > 14.0:
            raise PredictionError(f"pH level must be between 0 and 14. Got: {ph_val}")
            
        # 6. Humidity validation (percentage)
        humidity_val = validated_data['humidity']
        if humidity_val > 100.0:
            raise PredictionError(f"Humidity cannot exceed 100%. Got: {humidity_val}")
            
        return validated_data

    def predict(self, data: dict) -> str:
        """
        Receives user inputs, validates them, pre-processes, and returns crop prediction.
        """
        # Load artifacts if not loaded yet
        if self.model is None or self.scaler is None or self.label_encoder is None:
            self.load_artifacts()

        # Validate inputs
        validated = self.validate_inputs(data)
        
        # Convert to DataFrame with correct column ordering
        input_df = pd.DataFrame([validated])[self.feature_cols]
        
        try:
            # Scale features
            scaled_features = self.scaler.transform(input_df)
            scaled_features_df = pd.DataFrame(scaled_features, columns=self.feature_cols)
            
            # Predict
            pred_encoded = self.model.predict(scaled_features_df)
            
            # Decode crop label
            crop_name = self.label_encoder.inverse_transform(pred_encoded)[0]
            logger.info(f"Inputs: {validated} => Predicted Crop: {crop_name}")
            return crop_name
        except Exception as e:
            raise PredictionError(f"Error during prediction execution: {str(e)}")
