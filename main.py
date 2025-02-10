from dlProject import logger
from dlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline, DataTransformationPipeline, DataSplitPipeline
# from dlProject.pipeline.stage_02_data_transformation import DataTransformationPipeline

STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> stage Data Ingestion started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage Data Ingestion completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

try:
    logger.info(f">>>>>> stage Data Transformation started <<<<<<")
    obj = DataTransformationPipeline()
    obj.main()
    logger.info(f">>>>>> stage Data Transformation completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

try:
    logger.info(f">>>>>> stage Data Split started <<<<<<")
    obj = DataSplitPipeline()
    obj.main()
    logger.info(f">>>>>> stage Data Split completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e