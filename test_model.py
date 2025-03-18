import pickle

# 📌 Load the saved model and vectorizer
with open("fake_news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# 📌 Function to test news
def predict_news(news_text):
    news_tfidf = vectorizer.transform([news_text])
    prediction = model.predict(news_tfidf)[0]
    return "FAKE NEWS" if prediction == "fake" else "REAL NEWS"

# 🔍 Test the model
while True:
    news = input("\nEnter a news statement (or type 'exit' to quit): ")
    if news.lower() == "exit":
        break
    print(f"📰 Prediction: {predict_news(news)}")
