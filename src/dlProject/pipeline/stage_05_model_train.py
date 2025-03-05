from pathlib import Path
from dlProject import logger

from dlProject.commons.create_tf_dataset import create_train_test_dataset_tf
from dlProject.config.configuration import ConfigurationManager
from dlProject.components.model_train import TrainModelDl


class TrainModelDlPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        train_model_dl_config = config_manager.get_train_model_dl_config()
        train_dataset, val_dataset = create_train_test_dataset_tf(
            data_file=Path(train_model_dl_config.data_source_dir, train_model_dl_config.train_data_file_name),
            params=train_model_dl_config.params,
            train=True,
            evaluation=False
        )
        logger.info(f"\nTF Train dataset created")
        train_model_dl = TrainModelDl(config=train_model_dl_config, train_dataset=train_dataset, val_dataset=val_dataset)
        model = train_model_dl.train_model_dl()
        
        """ Save model """
        # model_h5_path = Path(train_model_dl_config.root_dir, f"{train_model_dl_config.train_model_file_name}")
        # model.save(model_h5_path)
        # logger.info(f"\nModel saved to {model_h5_path}")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage Model Train started <<<<<<")
        obj = TrainModelDlPipeline()
        obj.main()
        logger.info(f">>>>>> stage Model Train completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e