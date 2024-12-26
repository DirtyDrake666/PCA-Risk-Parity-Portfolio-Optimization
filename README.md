### 1. **Overview**
This script implements a portfolio optimization model based on Principal Component Analysis (PCA) and Risk Parity strategy. By analyzing historical asset data, it uses PCA for dimensionality reduction and Risk Parity for portfolio weight optimization. Key components:
- **Data Loading & Preprocessing**: Loads daily log returns and closing prices, aggregates quarterly data
- **Risk-Return Metrics**: Calculates annualized returns, volatility, maximum drawdown, Sharpe ratio, Calmar ratio
- **Optimization Models**: Implements Risk Parity and PCA-based multi-objective optimization
- **Rolling Window Optimization**: Dynamically adjusts optimal weights using historical data
- **Output**: Exports optimal weights, risk-return metrics, and PCA eigenvalues to CSV files

### 2. **Data Loading & Preprocessing**
Loads two main CSV files: daily log returns and closing prices. Data is aggregated quarterly to identify last trading days. Uses 2009 as base period and applies PCA for dimensionality reduction into 7 principal components.
```python
df = pd.read_csv('../data/daily_ln_index_yield.csv', index_col='time')
df_closing = pd.read_csv('../data/daily_closing_price.csv', index_col='time')
df1['Qtr'] = pd.PeriodIndex(pd.to_datetime(df1['time']), freq='Q')
grouped = df1['time'].groupby(df1['Qtr']).tail(1)
```

### 3. **Model Metrics**
Uses functions to calculate key portfolio metrics:
- **Cumulative Returns**: Based on daily closing prices
- **Maximum Drawdown**: Calculates portfolio's historical maximum drawdown
- **Annualized Returns**, **Volatility**, **Sharpe Ratio**, **Calmar Ratio**, **Maximum Diversification Ratio**

### 4. **Optimization Models**
#### Risk Parity Model
Calculates Risk Contribution (RC) for each asset. Objective: equalize risk contributions across assets.
```python
def calculate_risk_contribution(weights):
    ...
def risk_budget_objective(weights):
    ...
```

#### Multi-objective Optimization
Combines Risk Parity with Sharpe Ratio optimization through `single_target` function.

### 5. **Rolling Window Calculation**
Uses rolling window method to update portfolio weights dynamically:
- Uses 252-day lookback period
- Recalculates optimal weights quarterly
- Stores PCA eigenvalues for each period

### 6. **Output Results**
Exports to CSV files:
- `pca_risk_parity_model_optimal_weights.csv`
- `pca_risk_parity_model_risk_return_metrics.csv`
- `pca_risk_parity_model_eigenvalues.csv`
- `pca_risk_parity_model_daily_prices.csv`

### 7. **Example Results**
Output includes average values for:
- Annualized Returns
- Annualized Volatility
- Maximum Drawdown
- Sharpe Ratio
- Calmar Ratio

### 8. **Conclusion**
Model combines PCA and Risk Parity to optimize portfolio risk exposure. Rolling window approach enables dynamic weight adjustment for market changes. Provides comprehensive risk-return analysis tool for asset management and investment decisions.
