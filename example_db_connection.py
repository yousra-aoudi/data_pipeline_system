import psycopg2
import pandas as pd
from sklearn.linear_model import LinearRegression

# Connect to the database
conn = psycopg2.connect(host="localhost", database="trades", user="user", password="password")

# Extract the data using an SQL query
query = "SELECT * FROM trade_execution_data"
df = pd.read_sql_query(query, conn)

# Preprocess the data
df = df.dropna()
df['trade_date'] = pd.to_datetime(df['trade_date'])

# Calculate summary statistics using SQL
query = "SELECT COUNT(*) FROM trade_execution_data"
num_trades = pd.read_sql_query(query, conn).iloc[0,0]

# Engineer features using Python
df['days_since_previous_trade'] = df['trade_date'].diff().dt.days
df['trades_per_day'] = df.groupby(df['trade_date'].dt.date)['trade_id'].transform('count')

# Build a model using Scikit-learn
X = df[['days_since_previous_trade', 'trades_per_day']]
y = df['trading_cost']
model = LinearRegression()
model.fit(X, y)

# Use the model to make predictions
predictions = model.predict(X)

# Determine the market impact of portfolio trades
impact = (predictions - y).mean()

# Use the insights and findings to inform portfolio management decisions
print("The average market impact of portfolio trades is", impact)
