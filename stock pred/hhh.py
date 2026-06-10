import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ==================================================
# CREATE FOLDER FOR GRAPHS
# ==================================================

if not os.path.exists("graphs"):
    os.makedirs("graphs")

# ==================================================
# DOWNLOAD TCS STOCK DATA
# ==================================================

print("Downloading TCS stock data...")

data = yf.download(
    "TCS.NS",
    start="2018-01-01",
    end="2025-01-01",
    auto_adjust=True,
    progress=False
)

# Fix latest yfinance MultiIndex issue

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print("\nDataset Shape:", data.shape)

# ==================================================
# STOCK PRICE VISUALIZATION
# ==================================================

plt.figure(figsize=(12,6))

plt.plot(data.index, data["Close"])

plt.title("TCS Closing Stock Price")
plt.xlabel("Date")
plt.ylabel("Price")

plt.grid(True)

plt.savefig("graphs/stock_price.png")
plt.close()

# ==================================================
# FEATURE ENGINEERING
# ==================================================

data["MA10"] = data["Close"].rolling(10).mean()

data["MA50"] = data["Close"].rolling(50).mean()

data["Volatility"] = data["Close"].rolling(10).std()

data["Lag1"] = data["Close"].shift(1)

data["Lag2"] = data["Close"].shift(2)

data["Lag3"] = data["Close"].shift(3)

data.dropna(inplace=True)

# ==================================================
# MOVING AVERAGE GRAPH
# ==================================================

plt.figure(figsize=(12,6))

plt.plot(data.index, data["Close"], label="Close")

plt.plot(data.index, data["MA10"], label="MA10")

plt.plot(data.index, data["MA50"], label="MA50")

plt.title("Moving Average Analysis")

plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()

plt.grid(True)

plt.savefig("graphs/moving_average.png")
plt.close()

# ==================================================
# RANDOM FOREST MODEL
# ==================================================

features = [
    "MA10",
    "MA50",
    "Volatility",
    "Lag1",
    "Lag2",
    "Lag3"
]

X = data[features]

y = data["Close"]

split = int(len(data) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

# ==================================================
# RANDOM FOREST GRAPH
# ==================================================

plt.figure(figsize=(12,6))

plt.plot(
    y_test.index,
    y_test,
    label="Actual"
)

plt.plot(
    y_test.index,
    rf_predictions,
    label="Predicted"
)

plt.title("Random Forest Prediction")

plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()

plt.grid(True)

plt.savefig("graphs/random_forest_prediction.png")
plt.close()

# ==================================================
# ARIMA MODEL
# ==================================================

train = data["Close"][:split]

test = data["Close"][split:]

arima_model = ARIMA(
    train,
    order=(5,1,0)
)

arima_fit = arima_model.fit()

arima_predictions = arima_fit.forecast(
    steps=len(test)
)

arima_mae = mean_absolute_error(
    test,
    arima_predictions
)

arima_rmse = np.sqrt(
    mean_squared_error(
        test,
        arima_predictions
    )
)

# ==================================================
# ARIMA GRAPH
# ==================================================

plt.figure(figsize=(12,6))

plt.plot(
    test.index,
    test,
    label="Actual"
)

plt.plot(
    test.index,
    arima_predictions,
    label="Predicted"
)

plt.title("ARIMA Forecasting")

plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()

plt.grid(True)

plt.savefig("graphs/arima_prediction.png")
plt.close()

# ==================================================
# RESULTS
# ==================================================

print("\n===================================")
print("TCS STOCK PREDICTION RESULTS")
print("===================================")

print("\nRandom Forest")

print("MAE :", round(rf_mae,2))
print("RMSE:", round(rf_rmse,2))

print("\nARIMA")

print("MAE :", round(arima_mae,2))
print("RMSE:", round(arima_rmse,2))

print("\n===================================")

if rf_rmse < arima_rmse:
    print("Random Forest performed better.")
else:
    print("ARIMA performed better.")

print("\nGraphs saved in graphs folder.")

print("\nProject completed successfully.")