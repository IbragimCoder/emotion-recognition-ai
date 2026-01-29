# AI Emotion Recognition System

**Live Demo:**  https://emotion-recognition-ai-dk3zqgbsee9rzx6g8ewes5.streamlit.app/

#
#


![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)



This repository contains a **Machine Learning pipeline** and an interactive **Web Application** for Sentiment Analysis. The goal of the project is to classify text into one of six basic emotions: *sadness, joy, love, anger, fear, surprise*.



This project was developed as part of an **AI Internship technical assignment**.



---



## Features

* **Text Preprocessing:** Custom cleaning pipeline (lowercase, punctuation removal, noise reduction).

* **ML Model:** TF-IDF Vectorization + Logistic Regression with balanced class weights.

* **Interactive UI:** A user-friendly web interface built with **Streamlit**.

* **Robust Input Validation:**

    * Filters out non-English characters.

    * Rejects meaningless text (gibberish).

* **Uncertainty Handling:** The model admits uncertainty if the prediction confidence is below **40%**, preventing random guesses on out-of-distribution data.



---



## Installation & Usage



### 1. Clone the repository


git clone [https://github.com/IbragimCoder/emotion-recognition-ai.git](https://github.com/IbragimCoder/emotion-recognition-ai.git)
cd emotion-recognition-ai

### 2. Install dependencies

pip install -r requirements.txt


### 3. Run the Web App

streamlit run app.py

The application will open in your browser at http://localhost:8501.



### 4. Retrain the Model (Optional)

If you want to regenerate the model file emotion_model.joblib from scratch:


python create_model.py

## Model Analysis & Theory

For this multiclass classification task, I chose a classical Machine Learning approach (TF-IDF + Logistic Regression) over Deep Learning (BERT/RNN) for the following reasons:



Efficiency: Training takes seconds, and inference is near-instant (low latency), making it ideal for real-time applications.



Dataset Size: With 16,000 samples, classical ML provides a strong baseline without the risk of overfitting typical for large neural networks on small data.



Interpretability: Logistic Regression allows us to analyze feature importance (which words trigger which emotion).



Performance Evaluation



Overall Accuracy: 87%



Weighted F1-Score: 0.87



Weaknesses:



Love vs. Joy: There is some confusion between these classes due to semantic overlap (shared vocabulary like "happy", "good").



Surprise: This class has the lowest recall because it is underrepresented in the dataset (<3% of data). I addressed this by using class_weight='balanced', which significantly improved detection compared to the baseline.



📂 Project Structure

app.py — Main Streamlit application script (Web Interface).



create_model.py — Script to train the pipeline and save the .pkl file.



emotion_model.pkl — Serialized ML model (the "brain").



requirements.txt — List of Python dependencies.



training.csv / validation.csv / test.csv — Datasets.
