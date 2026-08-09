"""
03_train_model.py

Loads data/data.pickle, splits the dataset into 80% train and 20% test sets (stratified),
trains a scikit-learn RandomForestClassifier, evaluates accuracy, and saves the trained
model to models/model.p.
"""

import os
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Project root directory and paths definition
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PICKLE_PATH = PROJECT_ROOT / 'data' / 'data.pickle'
MODEL_DIR = PROJECT_ROOT / 'models'
MODEL_PATH = MODEL_DIR / 'model.p'


def train_model() -> None:
    """Trains a RandomForestClassifier on the hand landmark dataset and saves model weights."""
    if not os.path.exists(DATA_PICKLE_PATH):
        raise FileNotFoundError(
            f"Dataset pickle file not found at {DATA_PICKLE_PATH}. "
            "Please run '02_create_dataset.py' first."
        )

    with open(DATA_PICKLE_PATH, 'rb') as f:
        data_dict = pickle.load(f)

    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    score = accuracy_score(y_predict, y_test)
    print('{}% of samples were classified correctly !'.format(score * 100))

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model}, f)

    print(f"Model successfully saved to {MODEL_PATH}")


if __name__ == '__main__':
    train_model()
