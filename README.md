Mental Health Prediction Project
📌 Overview

This project explores text-based mental health prediction using machine learning and NLP techniques.
We preprocess raw mental health data, extract features (TF-IDF), train models, and evaluate their performance.

The project is structured in a clean and modular way for reproducibility and scalability.


mental_health_project/          📂 (Main project folder / Git repo root)
│
│── data/                       📂 (all datasets stay here)
│   ├── raw/                    📂 (original dataset(s), not cleaned)
│   ├── processed/              📂 (cleaned/processed datasets)
│
│── notebooks/                  📂 (Jupyter notebooks for each stage)
│   ├── 01_data_cleaning.ipynb        📄
│   ├── 02_feature_extraction.ipynb   📄
│   ├── 03_model_training.ipynb       📄
│   ├── 04_evaluation.ipynb           📄
│   └── 05_agentic_ai_features.ipynb  📄
│
│── src/                        📂 (Python scripts, reusable code)
│   ├── data_preprocessing.py   📄
│   ├── feature_extraction.py   📄
│   ├── ml_model.py             📄
│   └── utils.py                📄
│
│── results/                    📂 (graphs, metrics, outputs)
│
│── static/                     📂 (CSS, JS, images for web app)
│
│── templates/                  📂 (HTML files for web app)
│
│── app.py                      📄 (Main application entry point)
│── requirements.txt            📄 (dependencies list)
│── .gitignore                  📄 (files/folders to ignore in git)
│── README.md                   📄 (project documentation)
