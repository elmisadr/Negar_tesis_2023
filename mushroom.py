import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tsetlinMachine import TsetlinMachine
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np



# Define the URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

# Define the column names
col_names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
             'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 
             'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 
             'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 
             'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

# Load the data
df = pd.read_csv(url, header=None, names=col_names)

# Instantiate the encoders
label_enc = LabelEncoder()
one_hot_enc = OneHotEncoder()

# Encode target variable
df['class'] = label_enc.fit_transform(df['class'])

# One-hot encode the categorical features
df_encoded = pd.get_dummies(df)

# Separate features and target
X = df_encoded.drop('class', axis=1).values
y = df_encoded['class'].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize your TsetlinMachine
tm = TsetlinMachine(number_of_clauses=10, number_of_features=X_train.shape[1], number_of_states=10, s=3.9, threshold=15)
alpha = tm.calculate_equation(s, decay_rate, epoch)
tm.fit(X_train, y_train, number_of_examples=X_train.shape[0], epochs=100, alpha=alpha)

# Predict on the training and test set
y_train_pred = np.array([tm.predict(x) for x in X_train])
y_test_pred = np.array([tm.predict(x) for x in X_test])

# Calculate training and test accuracies
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Calculate training and test precision
train_prec = precision_score(y_train, y_train_pred)
test_prec = precision_score(y_test, y_test_pred)

# Calculate training and test recall
train_recall = recall_score(y_train, y_train_pred)
test_recall = recall_score(y_test, y_test_pred)

# Print training and test accuracies
print(f'Training accuracy: {train_acc:.2f}, Precision: {train_prec:.2f}, Recall: {train_recall:.2f}')
print(f'Test accuracy: {test_acc:.2f}, Precision: {test_prec:.2f}, Recall: {test_recall:.2f}')

# Plot the training and test accuracy curves
plt.figure(figsize=(10, 6))
plt.plot(train_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.title('Train and Test Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
