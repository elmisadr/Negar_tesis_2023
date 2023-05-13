import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tsetlinMachine import TsetlinMachine
import matplotlib.pyplot as plt

# Load mushroom dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
mushroom_data = pd.read_csv(url, header=None)
mushroom_data = mushroom_data.sample(frac=1).reset_index(drop=True) # shuffle rows
X = mushroom_data.iloc[:, 1:]
y = mushroom_data.iloc[:, 0]

# One-hot encode X and y
enc = OneHotEncoder()
X = enc.fit_transform(X)
y = np.array([1 if i == 'p' else 0 for i in y])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Tsetlin Machine model
# Convert sparse matrix to dense matrix
X_train = X_train.toarray()

# Define Tsetlin Machine model
tm = TsetlinMachine(100, 2, 3.9,4,3)

# Train the model
epochs = 100
train_acc = []
val_acc = []
test_acc = []
for epoch in range(epochs):
    # Train on training set
    tm.fit(X_train, y_train, X_train.shape[0], epochs=100)
    train_accuracy = tm.evaluate(X_train, y_train, X_train.shape[0])
    train_acc.append(train_accuracy)
    
    # Evaluate on validation set
    val_accuracy = tm.evaluate(X_val, y_val, X_val.shape[0])
    val_acc.append(val_accuracy)
    
    print("Epoch {}: train accuracy = {}, val accuracy = {}".format(epoch+1, train_accuracy, val_accuracy))
    
# Evaluate the model on the test set
X_test = X_test.toarray() # convert sparse matrix to dense matrix
test_accuracy = tm.evaluate(X_test, y_test, X_test.shape[0])
print("Test accuracy = {}".format(test_accuracy))

   
    
# Plot the accuracy curves
plt.plot(train_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')
plt.legend()
plt.show()
