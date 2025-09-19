# app.py
from flask import Flask, render_template, request
from src.feature_extraction import extract_features
from src.utils import get_agentic_suggestions, load_model, predict
from pymongo import MongoClient
from datetime import datetime
import os

app = Flask(__name__)

# Load ML model once
model = load_model()

# Connect to MongoDB
client = MongoClient("mongodb+srv://samartha:root@mental-health-cluster.4ee1v2a.mongodb.net/?retryWrites=true&w=majority&appName=mental-health-cluster")
db = client["mental_health_db"]
collection = db["user_predictions"]

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def make_prediction():
    user_text = request.form.get("user_input", "")
    if not user_text.strip():
        return render_template("result.html", text=user_text, prediction="No input", suggestions=[])

    features = extract_features(user_text)
    prediction = predict(model, features)  # returns string
    suggestions = get_agentic_suggestions(prediction)

     #4. Save everything to MongoDB
    record = {
        "text": user_text,
        "prediction": prediction,
        "suggestions": suggestions,   # ✅ now stored
        "timestamp": datetime.now()
    }
    collection.insert_one(record)

    return render_template("result.html", text=user_text, prediction=prediction, suggestions=suggestions)

@app.route("/history")
def history():
    records = list(collection.find().sort("timestamp", -1).limit(10))
    return render_template("history.html", records=records)

if __name__ == "__main__":
    app.run(debug=True)
