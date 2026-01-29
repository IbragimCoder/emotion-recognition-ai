import pandas as pd
import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

train_df = pd.read_csv('training.csv')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text
train_df['cleaned_text'] = train_df['text'].apply(clean_text)

X = train_df['cleaned_text']
y = train_df['label']
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
])

pipeline.fit(X, y)

joblib.dump(pipeline, 'emotion_model.joblib') 


print("The file 'emotion_model.joblib' created")