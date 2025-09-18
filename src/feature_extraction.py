# src/feature_extraction.py
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Global TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

def fit_vectorizer(texts):
    """
    Fit the TF-IDF vectorizer on training texts.
    This function ONLY fits in memory.
    Saving is handled outside this function.
    """
    global vectorizer
    vectorizer.fit(texts)

def transform_text(texts):
    """
    Transform texts to TF-IDF features using the fitted vectorizer.
    """
    return vectorizer.transform(texts)
