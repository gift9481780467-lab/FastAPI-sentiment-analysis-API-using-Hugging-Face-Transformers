from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="Sentiment Analysis API",
    description="Predicts sentiment using Hugging Face Transformers",
    version="1.0"
)

# Load sentiment model
sentiment_model = pipeline("sentiment-analysis")

# Input schema
class Review(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Sentiment API running"}

@app.post("/predict")
def predict(review: Review):

    if review.text.strip() == "":
        return {
            "error": "Review text cannot be empty"
        }

    result = sentiment_model(review.text)

    confidence = round(result[0]["score"] * 100, 2)

    return {
        "sentiment": result[0]["label"],
        "confidence": f"{confidence}%"
    }