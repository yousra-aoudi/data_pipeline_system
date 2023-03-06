"""
This class is designed to determine the market impact of portfolio trades by running a data pipeline to extract,
preprocess, and analyze trade execution data, and then using a linear regression model to make predictions about trading
costs. The determine_market_impact method calculates the average difference between the predicted and actual trading
costs, which can be used as a measure of market impact.
"""

from Data_Pipeline import DataPipeline
import pandas as pd
from sklearn.linear_model import LinearRegression


class MarketImpactAnalyzer:
    def __init__(self, host, database, user, password):
        self.pipeline = DataPipeline(host, database, user, password)

    def run_pipeline(self, query):
        self.pipeline.extract_data(query)
        self.pipeline.preprocess_data()
        self.pipeline.engineer_features()
        self.pipeline.build_model(LinearRegression)

    def determine_market_impact(self):
        predictions = self.pipeline.model.predict(self.pipeline.data[['days_since_previous_trade', 'trades_per_day']])
        return (predictions - self.pipeline.data['trading_cost']).mean()


