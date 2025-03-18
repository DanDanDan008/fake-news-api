import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# 📌 Load dataset
df = pd.read_csv("usa_disaster_fake_news.csv")

# 🔍 Text Cleaning Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Apply cleaning
df["text"] = df["text"].apply(clean_text)

# 📌 Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# 📌 Convert text to numerical form using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 📌 Save vectorizer
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

# 📌 Save processed data
train_data = {"X_train": X_train_tfidf, "y_train": y_train}
test_data = {"X_test": X_test_tfidf, "y_test": y_test}

with open("train_data.pkl", "wb") as train_file:
    pickle.dump(train_data, train_file)

with open("test_data.pkl", "wb") as test_file:
    pickle.dump(test_data, test_file)

print("✅ Data Cleaning & Preprocessing Done! Files saved.")
