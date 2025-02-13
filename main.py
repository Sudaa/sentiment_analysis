from model_trainer import evaluate_model
from huggingface_uploader import upload_model

if __name__ == "__main__":
    evaluate_model()
    upload_model()