from cnnClassifier import logger
from cnnClassifier.components.data_preparation import DataPreparation
from cnnClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Data Preparation"


class DataPreparationPipeline:
    def main(self):
        config = ConfigurationManager()
        DataPreparation(config=config.get_data_preparation_config()).run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        DataPreparationPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
