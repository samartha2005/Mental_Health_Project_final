# src/ml_model.py
import pickle
from sklearn.linear_model import LogisticRegression

# Global model object
model = LogisticRegression(max_iter=500)

def train_model(X, y):
    """
    Train the ML model on features X and labels y.
    Only fits in memory. Saving handled outside.
    """
    global model
    model.fit(X, y)

def load_model(path="src/models/trained_model.pkl"):
    """
    Load trained model from disk.
    """
    global model
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def predict(model, features):
    """
    Make predictions on features using the trained model.
    Returns a single string prediction.
    """
    return model.predict(features)[0]  # <-- return single value, not array
