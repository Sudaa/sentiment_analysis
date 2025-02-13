
import yaml
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


##load config
with open("config.yaml", "r") as file:
    config= yaml.safe_load(file)

def load_and_preprocess():
    '''loads adn preprocess dataset.'''
    dataset = load_dataset(config['dataset']['name'])

    # convert dataset to pandas DataFrame
    df = pd. DataFrame(dataset[config['dataset']['split']])

    # Ensure datset has 'text' and 'label' columns
    df = df.rename(columns={'text':'review', 'label':'sentiment'})

    # train-test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['review'], df['sentiment'],
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )

    #Tf-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, train_labels, test_labels, vectorizer

if __name__ == "__main__":
    X_train, X_test, train_labels, test_labels, vectorizer = load_and_preprocess()
    print('Dataset loaded and vectorized')

