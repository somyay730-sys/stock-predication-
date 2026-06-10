# 📈 Stock Market Prediction using Time Series Analysis

## Overview

This project aims to analyze and predict stock price trends of Tata Consultancy Services (TCS) using Time Series Analysis and Machine Learning techniques.

Historical stock data from Yahoo Finance was collected and processed to identify patterns and forecast future prices. Two predictive approaches were implemented:

* ARIMA (Time Series Model)
* Random Forest Regression (Machine Learning Model)

The performance of both models was evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

---

# Problem Statement

Stock prices exhibit trends, seasonality, volatility, and random fluctuations. Predicting future prices helps investors and analysts understand market behavior and make informed decisions.

This project investigates whether historical stock prices can be used to forecast future stock trends.

---

# Dataset

**Source:** Yahoo Finance

**Ticker:** TCS.NS

**Period:** 2018 – 2025

Dataset attributes:

* Open
* High
* Low
* Close
* Volume

---

# Feature Engineering

The following features were created:

### Moving Average (MA10)

Captures short-term trends.

### Moving Average (MA50)

Captures long-term trends.

### Volatility

Measures fluctuations in stock prices.

### Lag Variables

* Lag1
* Lag2
* Lag3

These represent previous closing prices and help models understand temporal dependency.

---

# Methodology

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. ARIMA Model Development
5. Random Forest Model Development
6. Model Evaluation
7. Visualization
8. Result Comparison

---

# Techniques Used

## Time Series Analysis

* ARIMA

## Machine Learning

* Random Forest Regression

## Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

---

# Visualizations

## Stock Trend

Insert image:

```markdown
Figure_1.png
```

---

## Moving Average Analysis

```markdown
![Moving Average](images/moving_average.png)
```

---

## Histogram of Closing Prices

```markdown
![Histogram](images/histogram.png)
```

---

## Daily Return Distribution

```markdown
![Daily Return](images/daily_return_histogram.png)
```

---

## Pie Chart: Gain vs Loss Days

```markdown
![Pie Chart](images/pie_chart.png)
```

---

## Box Plot

```markdown
![Box Plot](images/boxplot.png)
```

---

## Correlation Heatmap

```markdown
![Heatmap](images/heatmap.png)
```

---

## Random Forest Prediction

```markdown
![Random Forest](images/random_forest_prediction.png)
```

---

## ARIMA Prediction

```markdown
![ARIMA](images/arima_prediction.png)
```

---

## Model Comparison

```markdown
![Comparison](images/rmse_comparison.png)
```

---

# Results

| Model         |        MAE       |       RMSE       |
| ------------- |        -----     |       ----       |
| ARIMA         | 514.7119251442525| 602.0707435379039|
| Random Forest | 297.9831033325195| 387.9545796868183|

---

# Key Findings

* Historical prices significantly influence future stock prices.
* Feature engineering improved predictive performance.
* ARIMA effectively captured time-series trends.
* Random Forest handled nonlinear relationships and volatility more effectively.
* Combining statistical and machine learning approaches provides deeper insights into stock market behavior.

---

# Conclusion

This project demonstrates the application of Time Series Analysis and Machine Learning for stock price forecasting. The comparison between ARIMA and Random Forest highlights the strengths of both traditional statistical methods and modern machine learning techniques in understanding and predicting stock price movements.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Statsmodels
* yfinance

----
