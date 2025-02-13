import yaml
import joblib
import huggingface_hub

# Load config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def upload_model():
    """Uploads trained SVM model to Hugging Face."""
    huggingface_hub.login()

    # Load trained model & vectorizer
    model_path = f"{config['model']['output_dir']}/svm_model.joblib"
    vectorizer_path = f"{config['model']['output_dir']}/vectorizer.joblib"

    repo_name = f"{config['huggingface']['username']}/{config['huggingface']['repo_name']}"

    # Push model to Hugging Face Hub
    huggingface_hub.upload_file(path_or_fileobj=model_path, path_in_repo="svm_model.joblib", repo_id=repo_name)
    huggingface_hub.upload_file(path_or_fileobj=vectorizer_path, path_in_repo="vectorizer.joblib", repo_id=repo_name)

    print(f"✅ Model uploaded to Hugging Face: {repo_name}")

if __name__ == "__main__":
    upload_model()