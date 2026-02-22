# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from src.feature_extraction import extract_features
from src.utils import get_agentic_suggestions, load_model, predict
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# -------------------- SECURITY CONFIG --------------------

# Use environment variable in production
# Fallback keeps local development working
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey_dev")

# -------------------- LOAD MODEL --------------------

model = load_model()

# -------------------- DATABASE CONNECTION --------------------

MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://samartha:root@mental-health-cluster.4ee1v2a.mongodb.net/?retryWrites=true&w=majority&appName=mental-health-cluster"
)

client = MongoClient(MONGO_URI)
db = client["mental_health_db"]

predictions_collection = db["user_predictions"]
users_collection = db["users"]

# -------------------- HOME --------------------

@app.route("/")
def home():
    return redirect(url_for("login"))

# -------------------- REGISTER --------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        if not username or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("register"))

        existing_user = users_collection.find_one({"username": username})
        if existing_user:
            flash("Username already exists.", "error")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        users_collection.insert_one({
            "username": username,
            "password": hashed_password
        })

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# -------------------- LOGIN --------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        user = users_collection.find_one({"username": username})

        if user and check_password_hash(user["password"], password):
            session["user_id"] = str(user["_id"])
            session["username"] = user["username"]
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "error")
        return redirect(url_for("login"))

    return render_template("login.html")

# -------------------- DASHBOARD --------------------

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template("dashboard.html", username=session["username"])

# -------------------- LOGOUT --------------------

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))

# -------------------- PREDICT --------------------

@app.route("/predict", methods=["POST"])
def make_prediction():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_text = request.form.get("user_input", "").strip()

    if not user_text:
        flash("Please enter some text.", "error")
        return redirect(url_for("dashboard"))

    features = extract_features(user_text)
    prediction = predict(model, features)
    suggestions = get_agentic_suggestions(prediction)

    record = {
        "user_id": ObjectId(session["user_id"]),
        "username": session["username"],
        "text": user_text,
        "prediction": prediction,
        "suggestions": suggestions,
        "timestamp": datetime.now()
    }

    predictions_collection.insert_one(record)

    return render_template(
        "result.html",
        text=user_text,
        prediction=prediction,
        suggestions=suggestions
    )

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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))