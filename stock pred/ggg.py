
import tkinter as tk
from tkinter import ttk
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

# ---------------------- DATA ----------------------
data = yf.download("TCS.NS", start="2018-01-01", end="2025-01-01",
                   auto_adjust=True, progress=False)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data["MA10"] = data["Close"].rolling(10).mean()
data["MA50"] = data["Close"].rolling(50).mean()
data["Volatility"] = data["Close"].rolling(10).std()
data["Lag1"] = data["Close"].shift(1)
data["Lag2"] = data["Close"].shift(2)
data["Lag3"] = data["Close"].shift(3)
data["Daily_Return"] = data["Close"].pct_change() * 100
data.dropna(inplace=True)

# ---------------------- MODELS ----------------------
features = ["MA10","MA50","Volatility","Lag1","Lag2","Lag3"]
X = data[features]
y = data["Close"]

split = int(len(data)*0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

train = data["Close"][:split]
test = data["Close"][split:]

arima = ARIMA(train, order=(5,1,0))
arima_fit = arima.fit()
arima_pred = arima_fit.forecast(steps=len(test))

arima_mae = mean_absolute_error(test, arima_pred)
arima_rmse = np.sqrt(mean_squared_error(test, arima_pred))

# ---------------------- GUI ----------------------
root = tk.Tk()
root.title("TCS Stock Analysis Dashboard")
root.geometry("1400x900")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

tabs = {}
for name in ["Overview","Statistical Analysis","Feature Engineering",
             "Random Forest","ARIMA","Model Comparison"]:
    frame = ttk.Frame(notebook)
    notebook.add(frame, text=name)
    tabs[name] = frame

def add_chart(tab, fig):
    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# Overview
ttk.Label(tabs["Overview"],
          text=f"Latest TCS Price: ₹{data['Close'].iloc[-1]:.2f}",
          font=("Arial",14,"bold")).pack()

fig1 = plt.Figure(figsize=(8,4))
ax1 = fig1.add_subplot(111)
ax1.plot(data.index, data["Close"])
ax1.set_title("TCS Closing Price Trend")
add_chart(tabs["Overview"], fig1)

# Statistical Analysis
fig2 = plt.Figure(figsize=(10,8))

ax21 = fig2.add_subplot(221)
ax21.hist(data["Close"], bins=20)
ax21.set_title("Closing Price Histogram")

ax22 = fig2.add_subplot(222)
ax22.hist(data["Daily_Return"].dropna(), bins=30)
ax22.set_title("Daily Return Histogram")

ax23 = fig2.add_subplot(223)
ax23.boxplot(data["Close"])
ax23.set_title("Box Plot")

ax24 = fig2.add_subplot(224)
gain = (data["Daily_Return"] > 0).sum()
loss = (data["Daily_Return"] < 0).sum()
ax24.pie([gain, loss],
         labels=["Gain Days","Loss Days"],
         autopct="%1.1f%%")
ax24.set_title("Gain vs Loss Days")

add_chart(tabs["Statistical Analysis"], fig2)

# Feature Engineering
fig3 = plt.Figure(figsize=(10,8))

ax31 = fig3.add_subplot(211)
ax31.plot(data.index, data["Close"], label="Close")
ax31.plot(data.index, data["MA10"], label="MA10")
ax31.plot(data.index, data["MA50"], label="MA50")
ax31.legend()
ax31.set_title("Moving Averages")

ax32 = fig3.add_subplot(212)

corr_cols = ["Close","MA10","MA50","Volatility","Lag1","Lag2","Lag3"]
corr = data[corr_cols].corr()

if HAS_SEABORN:
    sns.heatmap(corr, annot=True, ax=ax32)
else:
    im = ax32.imshow(corr)
    ax32.set_xticks(range(len(corr.columns)))
    ax32.set_xticklabels(corr.columns, rotation=45)
    ax32.set_yticks(range(len(corr.columns)))
    ax32.set_yticklabels(corr.columns)

ax32.set_title("Correlation Heatmap")

add_chart(tabs["Feature Engineering"], fig3)

# Random Forest
ttk.Label(
    tabs["Random Forest"],
    text=f"MAE = {rf_mae:.2f} | RMSE = {rf_rmse:.2f}",
    font=("Arial",12,"bold")
).pack()

fig4 = plt.Figure(figsize=(8,4))
ax4 = fig4.add_subplot(111)
ax4.plot(y_test.index, y_test, label="Actual")
ax4.plot(y_test.index, rf_pred, label="Predicted")
ax4.legend()
ax4.set_title("Random Forest Prediction")
add_chart(tabs["Random Forest"], fig4)

# ARIMA
ttk.Label(
    tabs["ARIMA"],
    text=f"MAE = {arima_mae:.2f} | RMSE = {arima_rmse:.2f}",
    font=("Arial",12,"bold")
).pack()

fig5 = plt.Figure(figsize=(8,4))
ax5 = fig5.add_subplot(111)
ax5.plot(test.index, test, label="Actual")
ax5.plot(test.index, arima_pred, label="Predicted")
ax5.legend()
ax5.set_title("ARIMA Forecast")
add_chart(tabs["ARIMA"], fig5)

# Comparison
fig6 = plt.Figure(figsize=(8,5))
ax6 = fig6.add_subplot(111)

models = ["Random Forest","ARIMA"]
rmse = [rf_rmse, arima_rmse]

ax6.bar(models, rmse)
ax6.set_title("RMSE Comparison")
ax6.set_ylabel("RMSE")

add_chart(tabs["Model Comparison"], fig6)

root.mainloop()
