import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')

filename = os.path.join(os.path.dirname(__file__), 'abalone.csv')
df = pd.read_csv(filename)

df_male = df[df['Sex'] == 'M']
df_female = df[df['Sex'] == 'F']
df_infant = df[df['Sex'] == 'I']

fig1, male_axs = plt.subplots(2, 4, figsize=[13, 5])
fig1.suptitle('Male mono-variate histograms')

fig2, female_axs = plt.subplots(2, 4, figsize=[13, 5])
fig2.suptitle('Female mono-variate histograms')

fig3, infant_axs = plt.subplots(2, 4, figsize=[13, 5])
fig3.suptitle('Infant mono-variate histograms')


def histogram_plot(data, subp, color):
    print(data.name, ':', sep='')
    print("Média:", data.mean())
    print("Desvio padrão:", data.std())
    print("Assimetria:", data.skew(), "\n")
    subp.hist(data, alpha=0.5, color=color)
    subp.set_xlabel(data.name)
    subp.set_ylabel("Frequency")


def predictors_histogram_plot(data, axs, color):
    histogram_plot(data["Length"], axs[0, 0], color)
    histogram_plot(data["Diameter"], axs[0, 1], color)
    histogram_plot(data["Height"], axs[0, 2], color)
    histogram_plot(data["Whole weight"], axs[0, 3], color)
    histogram_plot(data["Shucked weight"], axs[1, 0], color)
    histogram_plot(data["Viscera weight"], axs[1, 1], color)
    histogram_plot(data["Shell weight"], axs[1, 2], color)
    histogram_plot(data["Rings"], axs[1, 3], color)


print("Male:")
predictors_histogram_plot(df_male, male_axs, 'b')
fig1.tight_layout()

print("Female:")
predictors_histogram_plot(df_female, female_axs, 'm')
fig2.tight_layout()

print("Infant:")
predictors_histogram_plot(df_infant, infant_axs, 'y')
fig3.tight_layout()

plt.show()
