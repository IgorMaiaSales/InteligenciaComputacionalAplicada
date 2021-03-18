import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

plt.style.use('ggplot')

filename = os.path.join(os.path.dirname(__file__), 'abalone.csv')
df = pd.read_csv(filename)

df_male = df[df['Sex'] == 'M']
df_female = df[df['Sex'] == 'F']
df_infant = df[df['Sex'] == 'I']

# Define preditor que deve ser plotado
predictor = "Length"

fig1 = plt.figure(figsize=(8, 8))
ax1 = fig1.add_subplot(1, 1, 1)

# Modificar intervalo de acordo com o preditor que está sendo plotado
bins = np.linspace(0, 1, 10)
ax1.hist(df_male[predictor], bins, alpha=0.5, color='b')
ax1.hist(df_female[predictor], bins, alpha=0.5, color='m')
ax1.hist(df_infant[predictor], bins, alpha=0.5, color='y')
# Modificar legenda de acordo com o preditor que está sendo plotado
fig1.suptitle('Class-conditional Length histograms')
targets = ['M', 'F', 'I']
ax1.legend(targets)

fig2 = plt.figure(figsize=(8, 8))
ax2 = fig2.add_subplot(1, 1, 1)

ax2.hist(df[predictor], alpha=0.5)
# Modificar legenda de acordo com o preditor que está sendo plotado
fig2.suptitle('Unconditional Length histogram')

plt.show()
