from wine_quality_project import logger
from wine_quality_project.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os

import pandas as pd

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)
        train, test = train_test_split(data, test_size=0.2, random_state=42)

        logger.info("Splitted data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)
        return train, test
    
    def preprocess(self, train, test):
        cols = train.columns[:-1]
        print(cols)
        target = self.config.target
        pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
        )
        preprocessor = ColumnTransformer([('pipeline', pipeline, cols)])
        X_train, X_test = train.drop([target],axis=1), test.drop([target],axis=1)
        y_train, y_test = train[[target]], test[[target]]
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        train = pd.DataFrame(X_train, columns=cols)
        train[target] = y_train.values
        test = pd.DataFrame(X_test, columns=cols)
        test[target] = y_test.values
        return preprocessor,train, test
    
    def save2artifact(self, preprocessor, train, test):
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
        joblib.dump(preprocessor, os.path.join(self.config.root_dir, self.config.preprocessor_name))


    