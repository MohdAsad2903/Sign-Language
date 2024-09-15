import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize the MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the directory where the data is located
DATA_DIR = './data'

# Initialize lists to store data and labels
data = []
labels = []

# Loop through each directory (class) in the data directory
for dir_ in os.listdir(DATA_DIR):
    # Loop through each image file in the current directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # Initialize a list to store hand landmark data for the current image
        data_aux = []

        # Initialize lists to store x and y coordinates of landmarks
        x_ = []
        y_ = []

        # Read the image using OpenCV
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))

        # Convert the image to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with the MediaPipe Hands model
        results = hands.process(img_rgb)

        # Check if hand landmarks are detected in the image
        if results.multi_hand_landmarks:
            # Loop through each detected hand landmark
            for hand_landmarks in results.multi_hand_landmarks:
                # Loop through each landmark point and collect x and y coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalize the coordinates relative to the minimum values
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Append the collected data for this image to the data list
            data.append(data_aux)

            # Append the class label (directory name) to the labels list
            labels.append(dir_)

# Save the collected data and labels to a pickle file
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
