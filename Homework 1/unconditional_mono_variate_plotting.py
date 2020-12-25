import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')

filename = os.path.join(os.path.dirname(__file__), 'abalone.csv')
df = pd.read_csv(filename)

fig, axs = plt.subplots(2, 4, figsize=[13, 5])


def unconditional_mono_variate(data, subp):
    print(data.name, ':', sep='')
    print("Média:", data.mean())
    print("Desvio padrão:", data.std())
    print("Assimetria:", data.skew())
    subp.hist(data, alpha=0.5)
    subp.set_xlabel(data.name)
    subp.set_ylabel("Frequency")


unconditional_mono_variate(df["Length"], axs[0, 0])
unconditional_mono_variate(df["Diameter"], axs[0, 1])
unconditional_mono_variate(df["Height"], axs[0, 2])
unconditional_mono_variate(df["Whole weight"], axs[0, 3])
unconditional_mono_variate(df["Shucked weight"], axs[1, 0])
unconditional_mono_variate(df["Viscera weight"], axs[1, 1])
unconditional_mono_variate(df["Shell weight"], axs[1, 2])
unconditional_mono_variate(df["Rings"], axs[1, 3])

fig.tight_layout()
plt.show()
