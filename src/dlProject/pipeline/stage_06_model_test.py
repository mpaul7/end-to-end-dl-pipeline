import pandas as pd
from pathlib import Path
from dlProject import logger

from dlProject.commons.create_tf_dataset import create_train_test_dataset_tf
from dlProject.utils.classification_report import getClassificationReport
from dlProject.config.configuration import ConfigurationManager
from dlProject.components.model_test import TestModelDl
from dlProject.utils.bargraph import plot_dynamic_bargraph
class TestModelDlPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        test_model_dl_config = config_manager.get_test_model_dl_config()
        
        test_files = test_model_dl_config.test_files
        test_files = {
            "same_data": Path(test_model_dl_config.data_source_dir, test_files[0]),
            "SolanaTest_data": Path(test_model_dl_config.cross_data_source_dir, test_files[1]),
            "HomeOffice_data": Path(test_model_dl_config.cross_data_source_dir, test_files[2])
        }
        
        cms = {}

        for k, v in test_files.items():
        
            """Create test dataset"""
            test_dataset = create_train_test_dataset_tf(
                # data_file=Path(test_model_dl_config.data_source_dir, test_model_dl_config.test_data_file_name),
                data_file=v,
                params=test_model_dl_config.params,
                train=False,
                evaluation=True
            )
            test_model_dl = TestModelDl(config=test_model_dl_config, test_dataset=test_dataset)
            confusion_matrix = test_model_dl.test_model_dl()
            
            """Classification Report"""
            classification_report = getClassificationReport(
                _confusion_matrix=confusion_matrix,
                traffic_classes=test_model_dl_config.params.labels.target_labels)
            
            print("\n", classification_report)
            
            if 'applications' not in cms:
                cms['applications'] = classification_report.index
            cms[k] = classification_report.iloc[:, -1]
            confusion_matrix_file_name = Path(test_model_dl_config.root_dir, test_model_dl_config.model_file_name.split(".")[0] + f"_{k}_confusion_matrix.csv" )
            print("\n", classification_report)
            classification_report.to_csv(confusion_matrix_file_name)
            logger.info(f"\nClassification report saved to {confusion_matrix_file_name}")

        comparison_confusion_matrix_file_name = Path(test_model_dl_config.root_dir, test_model_dl_config.model_file_name.split(".")[0] + f"_{k}_comparison_confusion_matrix.csv" )
        merged_df = pd.DataFrame(cms)
        merged_df.to_csv(comparison_confusion_matrix_file_name)
        print(merged_df)
        graph_file_name = Path(test_model_dl_config.root_dir, test_model_dl_config.model_file_name.split(".")[0] + f"_{k}_comparison_confusion_matrix.png" )
        plot_dynamic_bargraph(merged_df, graph_file_name)
        logger.info(f"\nComparison confusion matrix saved to {graph_file_name}")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage Model Test started <<<<<<")
        obj = TestModelDlPipeline()
        obj.main()
        logger.info(f">>>>>> stage Model Test completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e