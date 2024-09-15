import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data and labels from the data.pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert data and labels to NumPy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize a Random Forest Classifier
model = RandomForestClassifier()

# Train the model on the training data
model.fit(x_train, y_train)

# Make predictions on the test data
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_predict, y_test)

# Print the accuracy score
print('{}% of samples were classified correctly !'.format(score * 100))

# Save the trained model to a pickle file
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
