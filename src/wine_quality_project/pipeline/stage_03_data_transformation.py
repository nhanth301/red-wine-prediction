from wine_quality_project.config.configuration import ConfigurationManager
from wine_quality_project.components.data_transformation import DataTransformation
from wine_quality_project import logger
from pathlib import Path

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"),"r") as f:
                status = f.read().split(" ")[-1]
            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation()
                data_transformation = DataTransformation(config=data_transformation_config)
                train, test = data_transformation.train_test_splitting()
                preprocessor, train, test = data_transformation.preprocess(train,test)
                data_transformation.save2artifact(preprocessor,train,test)
            else:
                logger.exception(Exception("Your data schema is not valid"))
                raise Exception("Your data schema is not valid")
        except Exception as e:
            logger.exception(e)
            raise e