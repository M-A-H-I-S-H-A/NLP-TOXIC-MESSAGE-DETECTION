import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("train.csv")

# Select needed columns
data = data[['comment_text', 'toxic']]

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)   # remove URLs
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\d+", "", text)       # remove numbers
    return text

data['comment_text'] = data['comment_text'].apply(clean_text)

# Remove very short comments (improves accuracy)
data = data[data['comment_text'].str.split().str.len() > 2]

# Input & Output
X = data['comment_text']
y = data['toxic']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline (🔥 Optimized)
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        ngram_range=(1,3),   # better context
        min_df=2,
        max_df=0.9
    )),
    ('classifier', LogisticRegression(
        max_iter=300,
        C=2,
        class_weight='balanced'
    ))
])

# Train model
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("🔥 Accuracy:", round(accuracy, 4))

# Save model
pickle.dump(model, open("cyberbullying_model.pkl", "wb"))

print("✅ Model saved successfully!")