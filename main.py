# main.py

# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import datetime

# -------------------------
# 1. Define Parameters
# -------------------------
# List of asset tickers (you can replace these with your desired assets)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

# Define the time period for data download
start_date = '2018-01-01'
end_date = datetime.date.today().strftime('%Y-%m-%d')

# Risk-free rate (daily). For example, annual risk-free rate ~2% converted to daily
risk_free_rate = 0.02 / 252

# -------------------------
# 2. Data Collection & Preprocessing
# -------------------------
# Download historical data
data = yf.download(tickers, start=start_date, end=end_date)['Close']

# Compute daily returns
returns = data.pct_change().dropna()

# Plot historical price data for reference
data.plot(figsize=(12, 6), title='Asset Adjusted Close Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# -------------------------
# 3. Machine Learning for Return Prediction
# -------------------------
# For each asset, we will use a simple lag-1 model:
# Feature = previous day's return, Target = current day's return.
predicted_returns = {}

for ticker in tickers:
    # Create DataFrame with lag feature
    df = pd.DataFrame()
    df['return'] = returns[ticker]
    df['lag_return'] = df['return'].shift(1)
    df.dropna(inplace=True)
    
    # Prepare features (X) and target (y)
    X = df[['lag_return']].values
    y = df['return'].values
    
    # Split data into training and testing (here, simple split: 70% train, 30% test)
    split = int(0.7 * len(df))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Optionally, print model performance on test set
    score = model.score(X_test, y_test)
    print(f"{ticker} - Linear Regression R^2 on test data: {score:.3f}")
    
    # Use the most recent lag value to predict next day's return
    last_lag = df['lag_return'].iloc[-1]
    pred_return = model.predict(np.array([[last_lag]]))[0]
    predicted_returns[ticker] = pred_return

# Convert predicted returns into a NumPy array in the same order as tickers
predicted_return_array = np.array([predicted_returns[ticker] for ticker in tickers])
print("\nPredicted Next Day Returns:")
print(predicted_returns)

# -------------------------
# 4. Portfolio Optimization
# -------------------------
# Compute the covariance matrix from historical returns
cov_matrix = returns[tickers].cov().values

# Define a function to compute portfolio performance metrics
def portfolio_performance(weights, exp_returns, cov_matrix, risk_free_rate):
    port_return = np.dot(weights, exp_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (port_return - risk_free_rate) / port_vol if port_vol != 0 else 0
    return port_return, port_vol, sharpe_ratio

# Objective function: Negative Sharpe Ratio (to be minimized)
def neg_sharpe_ratio(weights, exp_returns, cov_matrix, risk_free_rate):
    _, port_vol, sharpe_ratio = portfolio_performance(weights, exp_returns, cov_matrix, risk_free_rate)
    return -sharpe_ratio

# Constraints: sum of weights must equal 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds: No short selling, weights between 0 and 1
bounds = tuple((0, 1) for _ in tickers)

# Initial guess: equally weighted portfolio
num_assets = len(tickers)
init_guess = np.repeat(1/num_assets, num_assets)

# Run the optimization
opt_result = minimize(neg_sharpe_ratio, init_guess, args=(predicted_return_array, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)

if opt_result.success:
    optimal_weights = opt_result.x
    port_return, port_vol, sharpe = portfolio_performance(optimal_weights, predicted_return_array, cov_matrix, risk_free_rate)
    print("\nOptimized Portfolio Weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight:.4f}")
    print(f"\nExpected Portfolio Return: {port_return:.4f}")
    print(f"Portfolio Volatility: {port_vol:.4f}")
    print(f"Portfolio Sharpe Ratio: {sharpe:.4f}")
else:
    print("Optimization failed.")

# -------------------------
# 5. (Optional) Visualize Portfolio Allocation
# -------------------------
plt.figure(figsize=(8, 4))
plt.bar(tickers, optimal_weights)
plt.title('Optimized Portfolio Allocation')
plt.xlabel('Asset')
plt.ylabel('Weight')
plt.show()



