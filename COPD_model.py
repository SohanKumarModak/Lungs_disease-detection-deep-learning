
# IMPORT REQUIRED LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


# LOAD DATASET

normal_df = pd.read_csv("ecg_dataset/ptbdb_normal.csv", header=None)
abnormal_df = pd.read_csv("ecg_dataset/ptbdb_abnormal.csv", header=None)

# ADD LABELS

normal_df[187] = 0
abnormal_df[187] = 1


# COMBINE DATASETS

dataset = pd.concat([normal_df, abnormal_df])


# SPLIT FEATURES AND LABELS

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


# NORMALIZE DATA

scaler = StandardScaler()
X = scaler.fit_transform(X)

# RESHAPE FOR CNN

X = X.reshape(X.shape[0], X.shape[1], 1)


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)


# BUILD 1D CNN MODEL
model = Sequential([

    Conv1D(32, 3, activation='relu', input_shape=(187,1)),
    MaxPooling1D(2),

    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),

    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])


# COMPILE MODEL

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# MODEL SUMMARY

model.summary()


# TRAIN MODEL

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32
)


# EVALUATE MODEL
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)


# PREDICTIONS

predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)


# CONFUSION MATRIX

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# CLASSIFICATION REPORT

print("Classification Report:")

print(classification_report(y_test, predictions))

# SAVE MODEL

model.save("copd_ecg_1dcnn_model.h5")

print("Model Saved Successfully")