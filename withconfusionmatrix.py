print('Importing libraries...')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn .datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

print('Loading...')
X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

print('extracting data from images...')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=3500, test_size=500)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)


# Confusion matrix
print('solving the matrix...')
samples_per_class = 5
figure = plt.figure(figsize=(nclasses*2, (1+samples_per_class*2)))

y_pred = clf.predict(X_test_scaled)
idx_cls = 0
for cls in classes:
	idxs = np.flatnonzero(y == cls)
	idxs = np.random.choice(idxs, samples_per_class, replace=False)
	i = 0
	for idx in idxs:
		plt_idx = i * nclasses + idx_cls + 1
		p = plt.subplot(samples_per_class, nclasses, plt_idx)
		p = sns.heatmap(np.reshape(X[idx], (22, 30)), cmap=plt.cm.gray, xticklabels=False, yticklabels=False, cbar=False)
		p = plt.axis("off")
		i += 1
	idx_cls += 1
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

print('done solving!')
print('now rendering...')
p = plt.figure(figsize=(10, 10))
p = sns.heatmap(cm, annot=True, fmt="d", cbar=False)

print('fetching the accuracy...')
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is :- ", accuracy)
