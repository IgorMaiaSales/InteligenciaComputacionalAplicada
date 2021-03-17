import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

reducedSet = pd.read_csv("reducedSet.csv")
testing = pd.read_csv("testing.csv")
training = pd.read_csv("training.csv")

X_train = training.filter(items=reducedSet.T.iloc[0 , :])
y_train = training.iloc[: , 1881]
X_test = testing.filter(items=reducedSet.T.iloc[0 , :])
y_test = testing.iloc[: , 1881]


sc_X = StandardScaler()
X_train2 = sc_X.fit_transform(X_train)
X_test2 = sc_X.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train2, y_train)

y_pred = logreg.predict(X_test2)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test2, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)