import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, classification_report

# 🔹 Load Dataset
df = pd.read_csv("usa_disaster_fake_news.csv")  # Replace with your actual dataset file

# 🔹 Check for missing values
df.dropna(inplace=True)

# 🔹 Feature & Label Selection
X = df["text"]  # Replace "text" with the actual text column name
y = df["label"]  # Replace "label" with 1 (real) and 0 (fake)

# 🔹 Split Data (Avoids Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# ✅ Add Noise to Training Data (Prevent Overfitting)
import random
def add_noise(text):
    words = text.split()
    if len(words) > 5:
        words[random.randint(0, len(words) - 1)] = "xyz"  # Replace a random word
    return " ".join(words)

X_train = X_train.apply(add_noise)  # Apply noise function


# 🔹 Vectorization (Convert Text to Features)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🔹 Handle Class Imbalance (Oversampling Fake News)
oversample = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train_vec, y_train = oversample.fit_resample(X_train_vec, y_train)

# 🔹 Fine-Tuned Logistic Regression Model
model = LogisticRegression(C=0.5, max_iter=500)  # Reduced C for better generalization
model.fit(X_train_vec, y_train)

# 🔹 Cross-Validation (More Reliable Accuracy)
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# 🔹 Predictions & Evaluation
y_pred = model.predict(X_test_vec)
print(f"Final Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 🔹 Save Model & Vectorizer
with open("fake_news_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✅ Model and Vectorizer Saved!")
