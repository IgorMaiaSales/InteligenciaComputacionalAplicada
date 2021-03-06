import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from IPython.display import display
# %matplotlib inline

filename = os.path.join(os.path.dirname(__file__), 'abalone.csv')
df = pd.read_csv(filename)
# display(df.head())

features = ['Length', 'Diameter', 'Height', 'Whole weight',
            'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
x = df.loc[:, features].values
y = df.loc[:, ['Sex']].values
x = StandardScaler().fit_transform(x)
#display(pd.DataFrame(data = x, columns = features).head())

varnames = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"]
pca = PCA()
principalComponents = pca.fit_transform(x)
explained_variance = pca.explained_variance_ratio_
plt.plot(varnames, explained_variance)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
#display(principalDf.head(5))
#display(df[['Sex']].head())
finalDf = pd.concat([principalDf, df[['Sex']]], axis = 1)
#display(finalDf.head(5))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)


targets = ['M', 'F', 'I']
colors = ['b', 'm', 'y']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Sex'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
ax.legend(targets)
ax.grid()
plt.show()
