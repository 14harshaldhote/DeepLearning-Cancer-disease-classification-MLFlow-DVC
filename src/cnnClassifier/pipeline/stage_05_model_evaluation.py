from cnnClassifier import logger
from cnnClassifier.components.model_evaluation import Evaluation
from cnnClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Evaluation"


class EvaluationPipeline:
    def main(self):
        config = ConfigurationManager()
        evaluation = Evaluation(config.get_evaluation_config())
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.promote_model()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        EvaluationPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
