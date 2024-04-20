import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pmdarima import auto_arima

# Load dataset
data = pd.read_csv('orientation_data.csv')

# Convert timestamps
data['time_stamp'] = pd.to_datetime(data['time_stamp'], unit='ms')

# Set the index to the timestamp
data.set_index('time_stamp', inplace=True)

# Prepare data for one angle and handle NaNs
x_angle_data = data['z_angle'].fillna(data['z_angle'].mean())

# Fit an ARIMA model
model = auto_arima(x_angle_data, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)

# Forecasting the next 100 steps
forecast, conf_int = model.predict(n_periods=100, return_conf_int=True)

# Create a time range for the forecasts
future_dates = pd.date_range(start=x_angle_data.index[-1], periods=101, freq='10L')[1:]

plt.figure(figsize=(100, 60))
plt.plot(x_angle_data.index, x_angle_data, label='Actual Values')
plt.plot(future_dates, forecast, label='Predicted Values', color='red')
plt.fill_between(future_dates, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.9)
plt.title('Forecast vs Actuals')
plt.legend()
plt.show()
