import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')

filename = os.path.join(os.path.dirname(__file__), 'abalone.csv')
df = pd.read_csv(filename)

fig, axs = plt.subplots(2, 4, figsize=[13, 5])
fig.suptitle('Unconditional mono-variate histograms')


def histogram_plot(data, subp):
    print(data.name, ':', sep='')
    print("Média:", data.mean())
    print("Desvio padrão:", data.std())
    print("Assimetria:", data.skew(), "\n")
    subp.hist(data, alpha=0.5)
    subp.set_xlabel(data.name)
    subp.set_ylabel("Frequency")


def predictors_histogram_plot(data, axs):
    histogram_plot(data["Length"], axs[0, 0])
    histogram_plot(data["Diameter"], axs[0, 1])
    histogram_plot(data["Height"], axs[0, 2])
    histogram_plot(data["Whole weight"], axs[0, 3])
    histogram_plot(data["Shucked weight"], axs[1, 0])
    histogram_plot(data["Viscera weight"], axs[1, 1])
    histogram_plot(data["Shell weight"], axs[1, 2])
    histogram_plot(data["Rings"], axs[1, 3])


predictors_histogram_plot(df, axs)
fig.tight_layout()
plt.show()
