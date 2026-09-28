"""Run the full training pipeline. `dvc repro` runs the same stages with caching."""

from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_data_preparation import DataPreparationPipeline
from cnnClassifier.pipeline.stage_03_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_trainer import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_05_model_evaluation import EvaluationPipeline

STAGES = [
    ("Data Ingestion", DataIngestionTrainingPipeline),
    ("Data Preparation", DataPreparationPipeline),
    ("Prepare Base Model", PrepareBaseModelTrainingPipeline),
    ("Training", ModelTrainingPipeline),
    ("Evaluation", EvaluationPipeline),
]


if __name__ == "__main__":
    for stage_name, pipeline in STAGES:
        try:
            logger.info(f">>>>>> stage {stage_name} started <<<<<<")
            pipeline().main()
            logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
        except Exception as e:
            logger.exception(e)
            raise e
