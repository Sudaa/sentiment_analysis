from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import huggingface_hub
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

# Load config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load the model and vectorizer from Hugging Face Hub
REPO_NAME = f"{config['huggingface']['username']}/{config['huggingface']['repo_name']}"
MODEL_PATH = "{config['model']['output_dir']}/svm_model.joblib"
VECTORIZER_PATH = "{config['model']['output_dir']}/vectorizer.joblib"


# Download model & vectorizer
huggingface_hub.hf_hub_download(repo_id=REPO_NAME, filename=MODEL_PATH, local_dir=".")
huggingface_hub.hf_hub_download(repo_id=REPO_NAME, filename=VECTORIZER_PATH, local_dir=".")

# Load them into memory
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Initialize FastAPI app
app = FastAPI()

# Define Pydantic model for request validation
class SentimentRequest(BaseModel):
    text: str

@app.post("/predict/")
def predict_sentiment(request: SentimentRequest):
    """Predict sentiment based on input text"""
    text_vectorized = vectorizer.transform([request.text])  # Convert input text to TF-IDF
    prediction = model.predict(text_vectorized)[0]  # Get prediction (0 = negative, 1 = positive)

    sentiment = "Positive" if prediction == 1 else "Negative"
    return {"text": request.text, "sentiment": sentiment}

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running!"}
