"""
The code below establishes a connection to a database, extracts trade execution data using an SQL query, preprocesses
the data, calculates summary statistics using SQL, engineers features using Python, builds a linear regression model 
using Scikit-learn, uses the model to make predictions, and determines the market impact of portfolio trades, and uses
the insights and findings to inform portfolio management decisions.
"""

import psycopg2
import pandas as pd
from sklearn.linear_model import LinearRegression


class DataPipeline:
    def __init__(self, host, database, user, password):
        self.summary_stats = None
        self.conn = psycopg2.connect(host=host, database=database, user=user, password=password)
        self.data = None
        self.model = None

    def extract_data(self, query):
        self.data = pd.read_sql_query(query, self.conn)

    def preprocess_data(self):
        self.data = self.data.dropna()
        self.data['trade_date'] = pd.to_datetime(self.data['trade_date'])

    def calculate_summary_statistics(self, query):
        self.summary_stats = pd.read_sql_query(query, self.conn)

    def engineer_features(self):
        self.data['days_since_previous_trade'] = self.data['trade_date'].diff().dt.days
        self.data['trades_per_day'] = self.data.groupby(self.data['trade_date'].dt.date)['trade_id'].transform('count')
        """
        This method adds two new features to the data: days_since_previous_trade, which is the number of days since the 
        previous trade, and trades_per_day, which is the number of trades per day.
        """

    def build_model(self, model_type):
        self.model = model_type()
        X = self.data[['days_since_previous_trade', 'trades_per_day']]
        y = self.data['trading_cost']
        self.model.fit(X, y)

    def evaluate_model(self, X=None):
        predictions = self.model.predict(X)
        return predictions.mean()
