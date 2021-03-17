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
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

reducedSet = pd.read_csv("reducedSet.csv")
testing = pd.read_csv("testing.csv")
training = pd.read_csv("training.csv")

X_train = training.filter(items=reducedSet.T.iloc[0 , :])
Y_train = training.iloc[: , 1881]
X_test = testing.filter(items=reducedSet.T.iloc[0 , :])
Y_test = testing.iloc[: , 1881]
