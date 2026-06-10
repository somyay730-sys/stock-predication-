
# STOCK MARKET ANALYSIS & PREDICTION - TCS
# ARIMA + RANDOM FOREST

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

#  DOWNLOAD STOCK DATA
print("Downloading TCS stock data...")
data = yf.download("TCS.NS",
                   start="2018-01-01",
                   end="2025-01-01")
print(data.head())

#  VISUALIZE STOCK PRICES
plt.figure(figsize=(12,6))
plt.plot(data['Close'])
plt.title("TCS Closing Stock Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# FEATURE ENGINEERING
# Moving Average
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
# Volatility
data['Volatility'] = data['Close'].rolling(window=10).std()
# Lag Features
data['Lag1'] = data['Close'].shift(1)
data['Lag2'] = data['Close'].shift(2)
data['Lag3'] = data['Close'].shift(3)
# Remove missing values
data.dropna(inplace=True)
print("\nFeature Engineered Data")
print(data.head())
#  TRAIN TEST SPLIT
features = ['MA10','MA50','Volatility','Lag1','Lag2','Lag3']
X = data[features]
y = data['Close']
split = int(len(data) * 0.8)
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

#  RANDOM FOREST MODEL
print("\nTraining Random Forest Model...")
rf_model = RandomForestRegressor(n_estimators=100,random_state=42)
rf_model.fit(X_train, y_train)
# Predictions
rf_predictions = rf_model.predict(X_test)

# RANDOM FOREST EVALUATION
rf_mae = mean_absolute_error(y_test,rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test,rf_predictions))
print("\nRANDOM FOREST RESULTS")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)

# RANDOM FOREST VISUALIZATION
plt.figure(figsize=(12,6))
plt.plot(y_test.index,y_test,label='Actual Price')
plt.plot(y_test.index,rf_predictions,label='RF Predicted Price')
plt.title("Random Forest Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

#ARIMA MODEL
print("\nTraining ARIMA Model...")
train = data['Close'][:split]
test = data['Close'][split:]
arima_model = ARIMA(train,order=(5,1,0))
arima_fit = arima_model.fit()
# Predict
arima_predictions = arima_fit.forecast(steps=len(test))
#ARIMA EVALUATION
arima_mae = mean_absolute_error(test,arima_predictions)
arima_rmse = np.sqrt(mean_squared_error(test,arima_predictions))
print("\nARIMA RESULTS")
print("MAE :", arima_mae)
print("RMSE:", arima_rmse)
# ARIMA VISUALIZATION
plt.figure(figsize=(12,6))
plt.plot(test.index,test,label='Actual Price')
plt.plot(test.index,arima_predictions,label='ARIMA Predicted Price')
plt.title("ARIMA Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
# MODEL COMPARISON
print("\n==========================")
print("MODEL COMPARISON")
print("==========================")
print("\nRandom Forest")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("\nARIMA")
print("MAE :", arima_mae)
print("RMSE:", arima_rmse)
