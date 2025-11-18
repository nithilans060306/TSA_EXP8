# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 18-11-2025


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

data = pd.read_csv('GoldPrice(2013-2023).csv')

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')

gold_data = data[['Date', 'Price']]
gold_data['Price'] = gold_data['Price'].astype(str).str.replace(',', '').astype(float)

plt.figure(figsize=(12, 6))
plt.plot(gold_data['Date'], gold_data['Price'], label='Original Gold Price')
plt.title('Original Gold Price Data')
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.legend()
plt.grid()
plt.show()

rolling_mean_5 = gold_data['Price'].rolling(window=5).mean()
rolling_mean_10 = gold_data['Price'].rolling(window=10).mean()

plt.figure(figsize=(12, 6))
plt.plot(gold_data['Date'], gold_data['Price'], label='Original Data', color='black')
plt.plot(gold_data['Date'], rolling_mean_5, label='Moving Average (window=5)', color='green')
plt.plot(gold_data['Date'], rolling_mean_10, label='Moving Average (window=10)', color='red')
plt.title('Moving Average of Gold Price')
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.legend()
plt.grid()
plt.show()

gold_data_monthly = gold_data.resample('MS', on='Date').mean()

scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(gold_data_monthly['Price'].values.reshape(-1, 1)).flatten(),
    index=gold_data_monthly.index
)

scaled_data = scaled_data + 1

split_index = int(len(scaled_data) * 0.8)
train_data = scaled_data[:split_index]
test_data = scaled_data[split_index:]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot(label='Train Data', figsize=(12,6))
test_predictions_add.plot(ax=ax, label='Predicted Data')
test_data.plot(ax=ax, label='Test Data')
ax.legend()
ax.set_title('Gold Price Prediction (Train vs Test)')
plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print("Root Mean Squared Error:", rmse)

model_full = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
future_predictions = model_full.forecast(steps=12)

ax = scaled_data.plot(label='Original Data', figsize=(12,6))
future_predictions.plot(ax=ax, label='Future Predictions')
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Scaled Gold Price')
ax.set_title('Gold Price Forecast (Next Year)')
plt.show()
```

### OUTPUT:
<img width="1140" height="631" alt="image" src="https://github.com/user-attachments/assets/6090b0cf-853c-448a-98a2-540b800d954f" />

<img width="1133" height="609" alt="image" src="https://github.com/user-attachments/assets/66ed2455-d102-45ad-87c5-aa1e7a0d5675" />

<img width="1136" height="619" alt="image" src="https://github.com/user-attachments/assets/f7573e89-1fbd-48ee-8696-817c6553a6f6" />

### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
