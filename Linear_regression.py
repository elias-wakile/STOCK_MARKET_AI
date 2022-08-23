import numpy as np
import pandas as pd
# Load local .csv file as DataFrame
from sklearn.model_selection import train_test_split

df = pd.read_csv('TSLA.csv')
df1 = pd.read_csv('TSLA.csv')
df1.set_index(pd.DatetimeIndex(df['Date']), inplace=True)

# Reindex data using a DatetimeIndex
df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)
# Keep only the 'Adj Close' Value
df = df[['Adj Close']]
# Re-inspect data

# print(df)
import pandas_ta

# Add EMA to dataframe by appending
# Note: pandas_ta integrates seamlessly into
# our existing dataframe
df.ta.ema(close='Adj Close', length=10, append=True)
# EMA_10.” This is our newly-calculated value representing the exponential moving average calculated over a 10-day period.
# print(df)
# print(df.info())

# print(df.head(10))
df = df.iloc[10:]
# print(df.head(10))


# Split data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(df[['Adj Close']], df[['EMA_10']], test_size=.2)
# Test set
# print(X_test.describe())
# print(X_train.describe())

from sklearn.linear_model import LinearRegression

# Create Regression Model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Use model to make predictions
y_pred = model.predict(X_test)

# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#
# # Printout relevant metrics
# print("Model Coefficients:", model.coef_)
# print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
# print("Coefficient of Determination:", r2_score(y_test, y_pred))

# print(y_pred[34])

print(X_test)
print('----')
print(y_pred)
print("+++++++++++++++++++++++++++++++++++++++")

# print(df1['Adj Close'][0]) ## get the index
# print(df1.index[0]) ## get the index
amount = 0
for j in range(len(X_test)):
    for i in range(len(df1)):
        if df1.index[i] == X_test.index[j]:
            if y_pred[j][0] - df1['Open'][i] > 0:
                print('date: ', df1.index[i],
                      '| Open: ', df1['Open'][i],
                      '| Close: ', df1['Close'][i],
                      '| predicted: ', y_pred[j],
                      '| gain: ', df1['Close'][i] - df1['Open'][i])
                amount += df1['Close'][i] - df1['Open'][i]

print(amount)

