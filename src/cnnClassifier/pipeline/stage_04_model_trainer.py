from cnnClassifier import logger
from cnnClassifier.components.model_trainer import Training
from cnnClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def main(self):
        config = ConfigurationManager()
        training = Training(config=config.get_training_config())
        training.get_base_model()
        training.train_valid_generator()
        training.train()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        ModelTrainingPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
