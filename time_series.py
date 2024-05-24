import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pmdarima import auto_arima

# Load dataset
data = pd.read_csv('orientation_data.csv')


data['time_stamp'] = pd.to_datetime(data['time_stamp'], unit='ms')


data.set_index('time_stamp', inplace=True)

# Prepare data for angles and handle NaNs
x_angle_data = data['x_angle'].fillna(data['x_angle'].mean())
y_angle_data = data['y_angle'].fillna(data['y_angle'].mean())
z_angle_data = data['z_angle'].fillna(data['z_angle'].mean())

# Fit ARIMA models for each angle
x_angle_model = auto_arima(x_angle_data, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
y_angle_model = auto_arima(y_angle_data, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
z_angle_model = auto_arima(z_angle_data, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)


x_angle_forecast, x_angle_conf_int = x_angle_model.predict(n_periods=100, return_conf_int=True)
y_angle_forecast, y_angle_conf_int = y_angle_model.predict(n_periods=100, return_conf_int=True)
z_angle_forecast, z_angle_conf_int = z_angle_model.predict(n_periods=100, return_conf_int=True)

# Create a time range for the forecasts
future_dates = pd.date_range(start=x_angle_data.index[-1], periods=101, freq='10L')[1:]


plt.figure(figsize=(16, 9))
plt.plot(x_angle_data.index, x_angle_data, label='Actual Values')
plt.plot(future_dates, x_angle_forecast, label='Predicted Values', color='red')
plt.fill_between(future_dates, x_angle_conf_int[:, 0], x_angle_conf_int[:, 1], color='pink', alpha=0.9)
plt.title('Forecast vs Actuals for X Angle')
plt.legend()
plt.show()


plt.figure(figsize=(16, 9))
plt.plot(y_angle_data.index, y_angle_data, label='Actual Values')
plt.plot(future_dates, y_angle_forecast, label='Predicted Values', color='red')
plt.fill_between(future_dates, y_angle_conf_int[:, 0], y_angle_conf_int[:, 1], color='pink', alpha=0.9)
plt.title('Forecast vs Actuals for Y Angle')
plt.legend()
plt.show()


plt.figure(figsize=(16, 9))
plt.plot(z_angle_data.index, z_angle_data, label='Actual Values')
plt.plot(future_dates, z_angle_forecast, label='Predicted Values', color='red')
plt.fill_between(future_dates, z_angle_conf_int[:, 0], z_angle_conf_int[:, 1], color='pink', alpha=0.9)
plt.title('Forecast vs Actuals for Z Angle')
plt.legend()
plt.show()
