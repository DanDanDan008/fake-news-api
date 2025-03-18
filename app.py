from fastapi import FastAPI
import pickle
from pydantic import BaseModel

# 📌 Load the trained model and vectorizer
with open("fake_news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# 📌 Create FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="API to detect fake news using machine learning",
    version="1.0.0"
)

# 📌 Define input data structure
class NewsItem(BaseModel):
    text: str

# 📌 Add home route
@app.get("/")
def home():
    return {"message": "Fake News API is running!"}

# 📌 Define predict API route
@app.post("/predict/")
def predict(news_item: NewsItem):
    news_tfidf = vectorizer.transform([news_item.text])
    prediction = model.predict(news_tfidf)[0]
    return {"prediction": "FAKE NEWS" if prediction == "fake" else "REAL NEWS"}

# 📌 Run with: uvicorn app:app --reload