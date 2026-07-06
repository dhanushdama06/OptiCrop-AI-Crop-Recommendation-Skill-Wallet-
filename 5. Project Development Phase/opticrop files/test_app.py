import unittest
import os
import json
from app import app
from models.prediction import CropPredictor, PredictionError

class TestOptiCrop(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.predictor = CropPredictor()
        
    def test_predictor_loading(self):
        """Test that predictor can load serialization artifacts."""
        self.predictor.load_artifacts()
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.scaler)
        self.assertIsNotNone(self.predictor.label_encoder)

    def test_valid_prediction(self):
        """Test prediction with standard, valid parameters."""
        # Standard values for rice
        data = {
            'N': '90',
            'P': '42',
            'K': '43',
            'temperature': '20.8',
            'humidity': '82.0',
            'ph': '6.5',
            'rainfall': '202.9'
        }
        crop = self.predictor.predict(data)
        self.assertEqual(crop, 'rice')

    def test_invalid_negative_values(self):
        """Test that negative parameters raise validation exceptions."""
        data = {
            'N': '-10', # Invalid negative
            'P': '42',
            'K': '43',
            'temperature': '20',
            'humidity': '82',
            'ph': '6.5',
            'rainfall': '202'
        }
        with self.assertRaises(PredictionError) as context:
            self.predictor.predict(data)
        self.assertIn("cannot be negative", str(context.exception))

    def test_invalid_ph_range(self):
        """Test that pH out of 0-14 range raises validation exceptions."""
        data = {
            'N': '90',
            'P': '42',
            'K': '43',
            'temperature': '20',
            'humidity': '82',
            'ph': '15.0', # Out of bounds
            'rainfall': '202'
        }
        with self.assertRaises(PredictionError) as context:
            self.predictor.predict(data)
        self.assertIn("pH level must be between 0 and 14", str(context.exception))

    def test_empty_values(self):
        """Test that empty string parameters raise validation exceptions."""
        data = {
            'N': '90',
            'P': '', # Empty
            'K': '43',
            'temperature': '20',
            'humidity': '82',
            'ph': '6.5',
            'rainfall': '202'
        }
        with self.assertRaises(PredictionError) as context:
            self.predictor.predict(data)
        self.assertIn("cannot be empty", str(context.exception))

    def test_non_numeric_values(self):
        """Test that string/non-numeric characters raise validation exceptions."""
        data = {
            'N': 'ninety', # Non-numeric
            'P': '42',
            'K': '43',
            'temperature': '20',
            'humidity': '82',
            'ph': '6.5',
            'rainfall': '202'
        }
        with self.assertRaises(PredictionError) as context:
            self.predictor.predict(data)
        self.assertIn("must be a valid number", str(context.exception))

    def test_flask_home_route(self):
        """Test that home route renders template successfully."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"OptiCrop", response.data)

    def test_flask_predict_route_valid(self):
        """Test that predict route processes POST data and returns recommended crop page."""
        response = self.app.post('/predict', data={
            'nitrogen': '90',
            'phosphorous': '42',
            'potassium': '43',
            'temperature': '20.8',
            'humidity': '82.0',
            'ph': '6.5',
            'rainfall': '202.9'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Recommended Crop", response.data)
        self.assertIn(b"rice", response.data.lower())

    def test_flask_predict_route_invalid(self):
        """Test that predict route flashes validation error on invalid input."""
        response = self.app.post('/predict', data={
            'nitrogen': '-10',
            'phosphorous': '42',
            'potassium': '43',
            'temperature': '20',
            'humidity': '82',
            'ph': '6.5',
            'rainfall': '202'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"cannot be negative", response.data)

if __name__ == "__main__":
    unittest.main()
