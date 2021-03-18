import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

filename = os.path.join(os.path.dirname(__file__), 'reducedSet.csv')
reducedSet = pd.read_csv(filename)

filename = os.path.join(os.path.dirname(__file__), 'testing.csv')
testing = pd.read_csv(filename)

filename = os.path.join(os.path.dirname(__file__), 'training.csv')
training = pd.read_csv(filename)

X_train = training.filter(items=reducedSet.T.iloc[0, :])
y_train = training.iloc[:, 1881]
X_test = testing.filter(items=reducedSet.T.iloc[0, :])
y_test = testing.iloc[:, 1881]

sc_X = StandardScaler()
X_train2 = sc_X.fit_transform(X_train)
X_test2 = sc_X.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
knn.fit(X_train2, y_train)

y_pred = knn.predict(X_test2)
print('Accuracy of K-Neighbors classifier on test set: {:.2f}'.format(
    knn.score(X_test2, y_test)))

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
