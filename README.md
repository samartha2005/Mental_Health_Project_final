🧠 MindCare AI
Intelligent Mental Health Prediction Web Application
📌 Overview

MindCare AI is a full-stack Mental Health Prediction Web Application built using Machine Learning, Natural Language Processing (NLP), Flask, and MongoDB.

The platform allows users to securely log in, submit text describing their mental state, receive AI-powered predictions, and view their personal prediction history through a clean and premium user interface.

This project demonstrates the integration of a trained ML model into a secure, production-ready web application.

🚀 Features
🔐 Secure Authentication System

User Registration & Login

Password hashing using Werkzeug

Secure session management

Protected routes (Dashboard & History)

Logout functionality

Flash messaging for user feedback

🧠 AI-Based Mental Health Prediction

Text preprocessing pipeline

TF-IDF feature extraction

Trained Machine Learning model (Scikit-learn)

Real-time prediction generation

Intelligent suggestions based on prediction

📊 User-Specific Prediction History

MongoDB Atlas cloud database

Separate collections for users and predictions

Timestamped records

User-based filtering (each user sees only their own history)

🎨 Premium UI Design

Clean and modern dashboard layout

Responsive design

Styled login and registration pages

Smooth user navigation

Organized result display

🛡 Security & Best Practices

Password hashing (no plain text passwords stored)

Session-based access control

Environment variable support for secret keys

Structured and modular codebase

Clean Git repository with .gitignore

🏗 Project Structure
mental_health_project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_agentic_ai_features.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── ml_model.py
│   └── utils.py
│
├── static/              # CSS, JS, images
├── templates/           # HTML templates
│
├── app.py               # Main Flask application
├── requirements.txt     # Dependencies
├── .gitignore
└── README.md
🛠 Tech Stack
🔹 Backend

Flask

MongoDB (PyMongo)

Gunicorn

🔹 Machine Learning

Scikit-learn

TF-IDF Vectorizer

NLTK

Pandas

NumPy

Joblib

🔹 Frontend

HTML

CSS

Jinja2 Templates

🧠 How MindCare AI Works

User registers and logs in securely.

User enters text describing their feelings.

Text is preprocessed and converted into TF-IDF features.

Machine Learning model predicts the mental health category.

Intelligent suggestions are generated.

Prediction is stored in MongoDB.

User can view their personal history anytime

💻 How to Run Locally
1️⃣ Clone the Repository
git clone <your-repository-url>
cd mental_health_project
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Application
python app.py

Open in browser:

http://127.0.0.1:5000