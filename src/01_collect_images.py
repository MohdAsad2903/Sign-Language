"""
01_collect_images.py

Collects image samples for 26 sign language alphabet classes via webcam.
Images are saved in class-specific subdirectories under data/.
"""

import os
from pathlib import Path
import cv2

# Project root directory and data path definition
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'


def collect_images(number_of_classes: int = 26, dataset_size: int = 100) -> None:
    """Captures dataset_size images per class for number_of_classes from the webcam."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    cap = cv2.VideoCapture(0)

    for j in range(number_of_classes):
        class_dir = os.path.join(DATA_DIR, str(j))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print('Collecting data for class {}'.format(j))

        while True:
            ret, frame = cap.read()
            cv2.putText(
                frame,
                'Ready? Press "Q" ! :)',
                (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.imshow('frame', frame)

            if cv2.waitKey(25) == ord('q'):
                break

        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
            counter += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    collect_images()
