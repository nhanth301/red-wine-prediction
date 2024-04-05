import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from wine_quality_project.config.configuration import ConfigurationManager

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.preprocessor = joblib.load(Path('artifacts/data_transformation/preprocessor.joblib'))

    def predict(self, data):
        cols = list(self.config.schema.COLUMNS.keys())
        data_df = pd.DataFrame(data, columns=cols[:-1])
        prediction = self.model.predict(self.preprocessor.transform(data_df))
        return prediction

