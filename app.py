# app.py
from flask import Flask, render_template, request, redirect, url_for, session
from src.feature_extraction import extract_features
from src.utils import get_agentic_suggestions, load_model, predict
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"   # session key

# Load ML model once
model = load_model()

# Connect to MongoDB
client = MongoClient("mongodb+srv://samartha:root@mental-health-cluster.4ee1v2a.mongodb.net/?retryWrites=true&w=majority&appName=mental-health-cluster")
db = client["mental_health_db"]

predictions_collection = db["user_predictions"]
users_collection = db["users"]


# -------------------- ROUTES --------------------

@app.route("/")
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


# -------------------- REGISTER --------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        existing_user = users_collection.find_one({"username": username})
        if existing_user:
            return "Username already exists"

        users_collection.insert_one({
            "username": username,
            "password": password
        })

        return redirect(url_for("login"))

    return render_template("register.html")


# -------------------- LOGIN --------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = users_collection.find_one({
            "username": username,
            "password": password
        })

        if user:
            session["user_id"] = str(user["_id"])
            session["username"] = user["username"]
            return redirect(url_for("home"))

        return "Invalid credentials"

    return render_template("login.html")


# -------------------- LOGOUT --------------------

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# -------------------- PREDICT --------------------

@app.route("/predict", methods=["POST"])
def make_prediction():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_text = request.form.get("user_input", "")

    if not user_text.strip():
        return render_template("result.html", text=user_text, prediction="No input", suggestions=[])

    features = extract_features(user_text)
    prediction = predict(model, features)
    suggestions = get_agentic_suggestions(prediction)

    # Save to MongoDB with user_id
    record = {
        "user_id": ObjectId(session["user_id"]),
        "text": user_text,
        "prediction": prediction,
        "suggestions": suggestions,
        "timestamp": datetime.now()
    }

    predictions_collection.insert_one(record)

    return render_template("result.html", text=user_text, prediction=prediction, suggestions=suggestions)


# -------------------- HISTORY --------------------

@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    records = list(
        predictions_collection.find(
            {"user_id": ObjectId(session["user_id"])}
        ).sort("timestamp", -1).limit(10)
    )

    return render_template("history.html", records=records)


# -------------------- RUN --------------------

if __name__ == "__main__":
    app.run(debug=True)
