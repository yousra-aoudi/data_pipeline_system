from Data_Pipeline import DataPipeline
import pandas as pd
from sklearn.linear_model import LinearRegression


class TradeCostPredictor:
    def __init__(self, host, database, user, password):
        self.pipeline = DataPipeline(host, database, user, password)

    def run_pipeline(self, query):
        self.pipeline.extract_data(query)
        self.pipeline.preprocess_data()
        self.pipeline.engineer_features()
        self.pipeline.build_model(LinearRegression)
