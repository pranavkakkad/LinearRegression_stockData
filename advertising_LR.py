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
pairplot_output = sns.pairplot(train_df, x_vars=['Open', 'High', 'Low', 'Adj Close', 'Volume'], y_vars = "Close", height=10, aspect=0.7)

pairplot_output.savefig("pairplot_output.png")

# from the graph volume does not matter

# creating model fit

lm1 = smf.ols(formula='Close ~ Open', data=train_df).fit()
# Printing the parameters
print(lm1.params)

feature_cols = ["Open"]
X = train_df[feature_cols]
y = train_df.Close

lm2 = LinearRegression()
lm2.fit(X, y)
print(lm2.intercept_)
print(lm2.coef_)


least_square_line = sns.pairplot(train_df, x_vars=['Open', 'High', 'Low', 'Adj Close', 'Volume'], y_vars = "Close", height=10, aspect= 0.7, kind="reg")
least_square_line.savefig("least_square_line.png")


lm1.conf_int()
lm1.pvalues

#Multiple Linear Regression

lm1 = smf.ols(formula='Close ~ Open + High + Low + Volume', data=train_df).fit()
print(lm1.params)


feature_cols=['Open', 'High', 'Low', 'Adj Close', 'Volume']
X = train_df[feature_cols]
y = train_df.Close

lm2 = LinearRegression()
lm2.fit(X, y)

print(lm2.intercept_)
print(lm2.coef_)

print(list(zip(feature_cols, lm2.coef_)))

print(lm1.summary())


# Feature Selection
# Volume have large p value

lm1 = smf.ols(formula="Close ~ Open + High + Low ", data=train_df).fit()
print(lm1.rsquared)

lm1 = smf.ols(formula="Close ~ Open + High + Low +Volume ", data=train_df).fit()
print(lm1.rsquared)



# Model Evaluation

X = train_df[["Open", "High", "Low", "Volume", "Adj Close"]]
y=train_df.Close

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
lm2 = LinearRegression()

lm2.fit(X_train, y_train)
y_pred = lm2.predict(X_test)

# RMSE
print("error with 'Volume' in Linear Regression ")
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

# Not considering Volume in Linear regression model

X = train_df[["Open", "High", "Low", "Adj Close"]]
y = train_df.Close
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
lm2 = LinearRegression()
lm2.fit(X_train, y_train)
y_pred = lm2.predict(X_test)

# RMSE
print("Error without considering 'Volume' feature in it")
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


