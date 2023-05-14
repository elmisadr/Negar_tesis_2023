import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tsetlinMachine import TsetlinMachine
import math
import scipy.stats as stats

# Load the iris dataset
iris = load_iris()

# Binarize the dataset using a threshold method
X = iris.data
y = iris.target
X = (X > 1).astype(np.int32)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the TsetlinMachine model
tm = TsetlinMachine(number_of_clauses=800, number_of_features=4, number_of_states=100, s=4, threshold=2)
epochs = 100
train_accs = []
test_accs = []

for epoch in range(epochs):
    np.random.shuffle(X_train)
    np.random.shuffle(y_train)
    for example_id in range(len(X_train)):
        target_class = y_train[example_id]
        Xi = X_train[example_id, :].astype(np.int32)
        tm.update(Xi, target_class, alpha)

# Calculate train and test accuracies
train_acc = tm.evaluate(X_train, y_train, len(X_train))
test_acc = tm.evaluate(X_test, y_test, len(X_test))
print(f"Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

# Calculate f1 score, precision, and recall
y_pred = [tm.predict(X_test[i]) for i in range(len(X_test))]
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"F1 score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

