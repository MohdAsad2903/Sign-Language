import os
import cv2

# Define the directory where data will be saved
DATA_DIR = './data'

# Create the data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes (26 for each alphabet letter)
number_of_classes = 26

# Number of samples to collect for each class
dataset_size = 100

# Open the webcam (camera index 0)
cap = cv2.VideoCapture(0)

# Loop through each class
for j in range(number_of_classes):
    # Create a directory for the current class if it doesn't exist
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    # Display a message indicating that the user should press 'Q' to start capturing data
    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Display a message to prompt the user to press 'Q'
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Check for the 'Q' key press to start capturing data
        if cv2.waitKey(25) == ord('q'):
            break

    # Initialize a counter to keep track of the number of samples collected
    counter = 0
    while counter < dataset_size:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Display the current frame
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save the captured frame as an image in the corresponding class directory
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        # Increment the counter
        counter += 1

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
