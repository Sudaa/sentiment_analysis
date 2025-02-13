
import yaml
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from data_loader import load_and_preprocess
##load config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def train_model():
    """Trains an SVC classifier for sentiment analysis"""
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess()

    ##train SVM Model
    model = SVC(kernel=config["model"]["kernel"], C=config["model"]["c"])
    model.fit(X_train, y_train)

    ##save the trained mdoel and vectorizer 
    joblib.dump(model, f"{config['model']['output_dir']}/svm_model.joblib")
    joblib.dump(vectorizer, f"{config['model']['output_dir']}/vectorizer.joblib")

    return model, X_test, y_test

def evaluate_model():
    """Evluate the trained model."""
    model, X_test, y_test = train_model()

    ##prediction
    y_pred = model.predict(X_test)

    ##Performance etris 
    accuracy= accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Model accuracy: {accuracy: .4f}")
    print(" Classification Report: \n ", report)

if __name__ == "__main__":
    evaluate_model()