from wine_quality_project import logger
from wine_quality_project.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from wine_quality_project.pipeline.statge_02_data_validation import DataValidationTrainingPipeline


STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<< \n\n x========================================x")
except  Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\n x========================================x")
except Exception as e:
    logger.exception(e)
    raise e