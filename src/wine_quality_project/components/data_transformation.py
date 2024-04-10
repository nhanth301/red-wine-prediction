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

        train_size = self.config.split_size.train
        val_size = self.config.split_size.val
        test_size = self.config.split_size.test

        train, test = train_test_split(data, test_size=(1-train_size), random_state=42)
        test, val = train_test_split(test, test_size=val_size/(val_size+test_size),random_state=42)

        logger.info("Splitted data into training, validation and test sets")
        logger.info(f'Train: {train.shape}')
        logger.info(f'Train: {val.shape}')
        logger.info(f'Train: {test.shape}')
        return train, val, test
    
    def preprocess(self, train, val, test):
        cols = train.columns[:-1]
        target = self.config.target
        pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
        )
        preprocessor = ColumnTransformer([('pipeline', pipeline, cols)])

        X_train, X_val, X_test = train.drop([target],axis=1), val.drop([target],axis=1), test.drop([target],axis=1)
        y_train, y_val, y_test = train[[target]], val[[target]], test[[target]]

        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)
        X_test = preprocessor.transform(X_test)
        
        train = pd.DataFrame(X_train, columns=cols)
        train[target] = y_train.values

        val = pd.DataFrame(X_val, columns=cols)
        val[target] = y_val.values

        test = pd.DataFrame(X_test, columns=cols)
        test[target] = y_test.values

        return preprocessor,train, val, test
    
    def save2artifact(self, preprocessor, train, val, test):
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        val.to_csv(os.path.join(self.config.root_dir, "val.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
        joblib.dump(preprocessor, os.path.join(self.config.root_dir, self.config.preprocessor_name))


    