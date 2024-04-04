# IMPORT LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# IMPORT DATA
from sklearn.datasets import load_digits
df = load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,3))
for ax, image,label in zip(axes, df.images, df.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# DATA PREPROCESSING
df.images.shape
df.images[0]
df.images[0].shape
len(df.images)

n_samples = len(df.images)
data = df.images.reshape((n_samples, -1))
data[0]
data[0].shape
data.shape

# SCALING IMAGE DATA
data.min()
data.max()
data = data/16
data.min()
data.max()
data[0]

# TRAIN TEST SPLIT DATA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, df.target, test_size = 0.3)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# RANDOM FOREST MODEL
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# PREDICT TEST DATA
y_pred = rf.predict(X_test)
y_pred

# MODEL ACCURACY
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
