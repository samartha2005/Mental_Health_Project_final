from flask import Flask, render_template, request
from pymongo import MongoClient
from datetime import datetime
import os
import pickle

# ---------------------------
# Helper functions to load model/vectorizer
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_vectorizer():
    path = os.path.join(BASE_DIR, "src", "models", "tfidf_vectorizer.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vectorizer file not found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_model_file():
    path = os.path.join(BASE_DIR, "src", "models", "trained_model.pkl")  # replace with your actual model filename
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------------------------
# Initialize Flask
# ---------------------------
app = Flask(__name__)

# Load ML model and vectorizer once
vectorizer = load_vectorizer()
model = load_model_file()

# Connect to MongoDB
client = MongoClient(
    "mongodb+srv://samartha:root@mental-health-cluster.4ee1v2a.mongodb.net/?retryWrites=true&w=majority&appName=mental-health-cluster"
)
db = client["mental_health_db"]
collection = db["user_predictions"]

# ---------------------------
# Preprocessing & prediction
# ---------------------------
def preprocess_text(text):
    # Replace with your actual preprocessing function from src.data_preprocessing
    return text.lower()  # simple placeholder

def extract_features(text):
    return vectorizer.transform([text])

def predict_model(model, features):
    return model.predict(features)[0]

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def make_prediction():
    user_text = request.form.get("user_input", "")
    if not user_text.strip():
        return render_template("result.html", text=user_text, prediction="No input provided")

    clean_text = preprocess_text(user_text)
    features = extract_features(clean_text)
    prediction = predict_model(model, features)

    # Save to MongoDB
    record = {
        "text": user_text,
        "prediction": prediction,
        "timestamp": datetime.now()
    }
    collection.insert_one(record)

    return render_template("result.html", text=user_text, prediction=prediction)

@app.route("/history")
def history():
    records = list(collection.find().sort("timestamp", -1).limit(10))
    return render_template("history.html", records=records)

# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
