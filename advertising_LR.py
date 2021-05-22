import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# importing the data
train_df = pd.read_csv("data/AAPL.csv")
print(train_df.shape)
print(train_df.columns)

# Visualise the relationship between features and and target using scatter plot

train_df = train_df.drop("Date", axis=1)
print(train_df.head())
# assuming target as closing price
sns.pairplot(train_df, x_vars=['Open', 'High', 'Low', 'Adj Close', 'Volume'], y_vars = "Close", height=10, aspect=0.7)
plt.show()