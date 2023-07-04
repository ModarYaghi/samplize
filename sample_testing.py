import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('sample63.csv')

# Variables of interest
variables = ['location', 'sex', 'age_group']

# Loop through each variable
for var in variables:
    print(f"---- {var} ----")

    # Select the sample data for this variable
    sample_data = df[df['sample'] == 1][var]

    # Histogram
    sns.histplot(sample_data, kde=True)
    plt.show()

    # Q-Q plot
    stats.probplot(sample_data, plot=plt)
    plt.show()

    # Shapiro-Wilk test
    shapiro_test = stats.shapiro(sample_data)
    print(f"Shapiro test statistic: {shapiro_test[0]}, p-value: {shapiro_test[1]}")

    # If p-value is less than 0.05, the data is not normally distributed
    if shapiro_test[1] < 0.05:
        print(f"{var} data is not normally distributed.")
        continue

    # Calculate the sample mean and the standard error of the mean
    sample_mean = np.mean(sample_data)
    sem = stats.sem(sample_data)

    # Calculate the 95% confidence interval
    confidence_interval = stats.t.interval(alpha=0.95, df=len(sample_data)-1, loc=sample_mean, scale=sem)

    print(f"The 95% confidence interval for the mean is {confidence_interval}")

    # Margin of error
    margin_of_error = confidence_interval[1] - sample_mean
    print(f"The margin of error is {margin_of_error}")
