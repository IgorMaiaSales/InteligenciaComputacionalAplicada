import pandas as pd
import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')

# Para usar relative path:
filename = os.path.join(os.path.dirname(__file__), 'abalone.csv')
df = pd.read_csv(filename)

# ========================================================================================
# Unconditional Mono-Variate
# Substituir 'Rings' pelo nome do preditor que se quer plotar (Manter as aspas)


def unconditional_mono_variate(data):
    print("Média:", data.mean())
    print("Desvio padrão:", data.std())
    print("Assimetria:", data.skew())
    data.plot.hist(alpha=0.5)
    plt.xlabel(data.name)
    plt.show()


unconditional_mono_variate(df["Rings"])

# ========================================================================================
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
# ========================================================================================
