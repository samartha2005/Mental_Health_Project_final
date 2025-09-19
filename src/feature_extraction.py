# src/feature_extraction.py
import pickle
import os
from src.data_preprocessing import preprocess_text

# Load trained TF-IDF vectorizer
VECTORIZER_PATH = os.path.join("src/models", "tfidf_vectorizer.pkl")
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

def extract_features(text: str):
    """
    Preprocess input text and transform it to TF-IDF features
    using the trained vectorizer.
    """
    cleaned_text = preprocess_text(text)
    return vectorizer.transform([cleaned_text])
