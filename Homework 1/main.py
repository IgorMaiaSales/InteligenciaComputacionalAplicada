import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

df = pd.read_csv("abalone.csv")

# =======================================================================================
# Unconditional Mono-Variate
# Subtituir 'Rings' pelo nome do preditor que se quer plotar (Manter as aspas)
print(df['Rings'].describe())
df['Rings'].plot.hist(alpha=0.5)
plt.xlabel('Rings')
plt.show()

# =======================================================================================
# Class Conditional Mono-Variate
# Sepração das Classes(Male, Female e Infant) em diferentes Data Frames
# df_male = df[df['Sex']=='M']
# df_female = df[df['Sex']=='F']
# df_infant = df[df['Sex']=='I']


# Subtituir 'Length' pelo nome do preditor que se quer plotar (Manter as aspas)
# df_male['Length'].plot.hist(alpha = 0.5,label='Male')
# df_female['Length'].plot.hist(alpha = 0.5,label='Female')
# df_infant['Length'].plot.hist(alpha = 0.5,label='Infant')
# plt.xlabel('Length')
# plt.legend(loc='best')
# plt.show()
# =====================================================================================