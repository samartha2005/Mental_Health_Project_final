# src/utils.py
from src.feature_extraction import extract_features
from src.ml_model import predict as ml_predict
import pickle
import os

# Load ML model
def load_model():
    model_path = "src/models/trained_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        model = None
    return model

# Prediction wrapper
def predict(model, features):
    """
    Wrapper around ml_model.predict that passes the model.
    Ensures a string is returned, not a NumPy array.
    """
    if model is None:
        return "Model not loaded"
    return ml_predict(model, features)

# Agentic AI suggestions
def get_agentic_suggestions(emotion: str):
    suggestions = {
        'Depression': [
            "Talk to a trusted friend or family member.",
            "Consider daily short walks or light exercise.",
            "Try online mindfulness exercises.",
            "Maintain a sleep routine."
        ],
        'Anxiety': [
            "Try deep breathing or meditation.",
            "Write down your worries.",
            "Limit caffeine and sugar.",
            "Do a short walk."
        ],
        'Stress': [
            "Take short breaks.",
            "Listen to calming music.",
            "Prioritize tasks and delegate.",
            "Try journaling or creative activities."
        ],
        'Neutral': [
            "Keep up positive habits!",
            "Balance work/study and leisure."
        ]
    }
    return suggestions.get(emotion, ["Consider consulting a mental health professional."])
