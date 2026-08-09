"""
02_create_dataset.py

Processes collected sign language images with MediaPipe Hands to extract 21 (x, y) 
hand landmark coordinates, normalizes landmarks relative to the hand's minimum x and y coordinates,
and saves the resulting feature vectors and class labels to data/data.pickle.
"""

import os
from pathlib import Path
import pickle

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

# Project root directory and paths definition
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
DATA_PICKLE_PATH = DATA_DIR / 'data.pickle'


def create_dataset() -> None:
    """Extracts MediaPipe hand landmarks from images under data/ and pickles the dataset."""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    data = []
    labels = []

    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' does not exist.")
        return

    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path):
            continue

        for img_path in os.listdir(dir_path):
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(dir_path, img_path))
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

    with open(DATA_PICKLE_PATH, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print(f"Dataset successfully created and saved to {DATA_PICKLE_PATH}")


if __name__ == '__main__':
    create_dataset()
