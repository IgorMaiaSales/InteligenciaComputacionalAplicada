import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.pylab as pylab

params = {'axes.labelsize': 'x-small',
          'axes.titlesize': 'x-small',
          'xtick.labelsize': 'x-small',
          'ytick.labelsize': 'x-small'}
pylab.rcParams.update(params)

filename = os.path.join(os.path.dirname(__file__), 'abalone.csv')
df = pd.read_csv(filename)

fig = sns.pairplot(df, hue='Sex', palette=['b', 'm', 'y'])
fig.fig.suptitle('Unconditional bi-variate analysis')
plt.show()
