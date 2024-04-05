from wine_quality_project.config.configuration import ConfigurationManager
from wine_quality_project.components.model_trainer import ModelTrainer
from wine_quality_project import logger

STAGE_NAME = "Model Trainer stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\n x========================================x")
    except Exception as e:
        logger.exception(e)
        raise e
