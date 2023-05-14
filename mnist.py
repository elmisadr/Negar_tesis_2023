import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from tsetlinMachine import TsetlinMachine
from tsetlinMachine import TsetlinMachine
import cv2


# Load MNIST dataset
mnist = fetch_openml('mnist_784')
X, y = mnist['data'], mnist['target'].astype(np.int32)

# Binarize the dataset using the Otsu thresholding method
binarized_X = np.zeros_like(X)
for i in range(X.shape[0]):
    _, binarized_X[i] = cv2.threshold(X[i], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(binarized_X, y, test_size=0.2, random_state=42)

# Train the Tsetlin Machine
number_of_examples = X_train.shape[0]
number_of_clauses = 8000
number_of_features = X_train.shape[1]
number_of_states = 100
s = 5
threshold = 2

tm = TsetlinMachine(number_of_clauses, number_of_features, number_of_states, s, threshold)
epochs = 500
alpha = tm.calculate_equation(s, decay_rate, epoch)
tm.fit(X_train, y_train, number_of_examples, epochs, alpha)

# Test the Tsetlin Machine and calculate performance metrics
y_pred_train = np.array([tm.predict(X_train[i]) for i in range(X_train.shape[0])])

y_pred_test = []
for i in range(X_test.shape[0]):
    print(f"Example {i+1}:")
    y_pred = tm.predict(X_test[i])
    y_pred_test.append(y_pred)


y_pred_test = np.array(y_pred_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)

# Plot train and test accuracies
plt.plot(train_accuracy, label="Train Accuracy")
plt.plot(test_accuracy, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

