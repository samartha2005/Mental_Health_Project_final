from src.data_preprocessing import preprocess_text
from src.feature_extraction import transform_text
from src.ml_model import predict as ml_predict
from src.ml_model import predict as ml_predict

import pickle
import os

# Feature extraction wrapper
def extract_features(text):
    """For single text input"""
    return transform_text([text])

# Load ML model
def load_model():
    """Return model instance (loads from pickle if exists)"""
    model_path = "src/ml_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        model = None  # Model not trained yet
    return model

# Prediction wrapper
def predict(model, features):
    if model is None:
        return "Model not loaded"
    return ml_predict(features)
