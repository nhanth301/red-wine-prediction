from wine_quality_project.entity.config_entity import ModelTrainerConfig
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from wine_quality_project.utils.common import create_mlflow_experiment,get_mlflow_experiment
from wine_quality_project.constants import EXPERIMENT_NAME
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import pandas as pd
import mlflow

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        val_data = pd.read_csv(self.config.val_data_path)

        X_train, X_val = train_data.drop([self.config.target_column], axis=1), val_data.drop([self.config.target_column], axis=1)
        y_train, y_val = train_data[[self.config.target_column]], val_data[[self.config.target_column]]

        experiment_id = create_mlflow_experiment(experiment_name=EXPERIMENT_NAME,artifact_location="mlflow_models",tags={})
        mlflow.set_experiment(experiment_id=experiment_id)
        models = {}
        for model, params in self.config.model_params.items():
            if model == "RandomForestRegressor":
                models[model] = (RandomForestRegressor(),params)
            if model == "ElasticNet":
                models[model] = (ElasticNet(),params)

        best_model = None
        best_score = float('-inf')
        run_id = None

        for model_name, (model, param_grid) in models.items():
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')   
            with mlflow.start_run(run_name=model_name) as run:
                grid_search.fit(X_train,y_train)
                best_model = grid_search.best_estimator_
                best_estimator = grid_search.best_estimator_
                score = grid_search.best_score_
                if score > best_score:
                    best_model = best_estimator
                    best_score = score
                    run_id = run.info.run_id
                mlflow.log_params(grid_search.best_params_)
                y_val_pred = best_estimator.predict(X_val)
                metrics = {
                    'mse' : mean_squared_error(y_val_pred,y_val),
                    'mae' : mean_absolute_error(y_val_pred,y_val),
                    'r2' : r2_score(y_val_pred,y_val)
                }
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(best_estimator, model_name + "_best_model")

        client = mlflow.MlflowClient()
        model_name = EXPERIMENT_NAME
        source = f"runs:/{run_id}"
        try:
            client.create_registered_model(model_name)
        except:
            pass
        client.create_model_version(name=model_name, source=source, run_id=run_id)
        joblib.dump(best_model, os.path.join(self.config.root_dir, self.config.model_name))