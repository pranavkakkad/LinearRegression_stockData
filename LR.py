import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

train_df = pd.read_csv("data/AAPL.csv")
# print(train_df`.shape)
print(train_df.columns)
train_df = train_df.drop(["Date", "Adj Close"], axis=1)
print(train_df.head())

# Splitting the data into training and test set
X = train_df.loc[:, train_df.columns != "Close"]
y = train_df.loc[:, train_df.columns == "Close"]

train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2)

print(train_x.head())
print(test_x.head())

reg = LinearRegression()
reg.fit(train_x,train_y)


predict_y = reg.predict(test_x)

print("Prediction score:", reg.score(test_x,test_y))

error = mean_squared_error(test_y,predict_y)
print("Mean square error", error)


fig = plt.figure()
ax = plt.axes()
ax.grid()
ax.set(xlabel = "Close", ylabel="Open", title="Apple stock")
ax.plot(test_x["Open"],test_y)
ax.plot(test_x["Open"],predict_y)
plt.show()