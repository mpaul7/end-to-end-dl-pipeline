from dlProject import logger
from dlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from dlProject.pipeline.stage_02a_data_preparation import DataPreparationPipeline
from dlProject.pipeline.stage_02a_data_preparation import DataPreparationPipeline
from dlProject.pipeline.stage_02_data_transformation import DataTransformationPipeline
from dlProject.pipeline.stage_03_data_split import DataSplitPipeline
from dlProject.pipeline.stage_03a_feature_selection import FeatureSelectionPipeline
from dlProject.pipeline.stage_04_model_build import ModelBuilderPipeline
from dlProject.pipeline.stage_05_model_train import TrainModelDlPipeline
from dlProject.pipeline.stage_06_model_test import TestModelDlPipeline

STAGE_NAME = "Data Ingestion stage"

# try:
#     logger.info(f">>>>>> stage Data Ingestion started <<<<<<")
#     obj = DataIngestionTrainingPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage Data Ingestion completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e

# try:
#     logger.info(f">>>>>> stage Data Transformation started <<<<<<")
#     obj = DataTransformationPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage Data Transformation completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e

# try:
#     logger.info(f">>>>>> stage Data Preparation started <<<<<<")
#     obj = DataPreparationPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage Data Preparation completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e

# try:
#     logger.info(f">>>>>> stage Data Split started <<<<<<")
#     obj = DataSplitPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage Data Split completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e

try:
    logger.info(f">>>>>> stage Feature Selection started <<<<<<")
    obj = FeatureSelectionPipeline()
    obj.main()
    logger.info(f">>>>>> stage Feature Selection completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# try:
#     logger.info(f">>>>>> stage Model Builder started <<<<<<")
#     obj = ModelBuilderPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage Model Builder completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e

# try:
#     logger.info(f">>>>>> stage Train Model DL started <<<<<<")
#     obj = TrainModelDlPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage Train Model DL completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e


# try:
#     logger.info(f">>>>>> stage Model Test started <<<<<<")
#     obj = TestModelDlPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage Model Test completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e