import os
from pathlib import Path

from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    DataPreparationConfig,
    EvaluationConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
)
from cnnClassifier.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(Path(config_filepath))
        self.params = read_yaml(Path(params_filepath))

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_source=Path(config.local_source),
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    def get_data_preparation_config(self) -> DataPreparationConfig:
        config = self.config.data_preparation
        create_directories([config.root_dir])

        return DataPreparationConfig(
            root_dir=Path(config.root_dir),
            source_dir=Path(config.source_dir),
            report_path=Path(config.report_path),
            params_val_split=self.params.VAL_SPLIT,
            params_test_split=self.params.TEST_SPLIT,
            params_seed=self.params.SEED,
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        split_root = Path(self.config.data_preparation.root_dir)
        create_directories([training.root_dir])

        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            history_path=Path(training.history_path),
            updated_base_model_path=Path(self.config.prepare_base_model.updated_base_model_path),
            train_dir=split_root / "train",
            val_dir=split_root / "val",
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_is_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
            params_patience=self.params.EARLY_STOPPING_PATIENCE,
            params_seed=self.params.SEED,
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        evaluation = self.config.evaluation
        create_directories([evaluation.root_dir])

        return EvaluationConfig(
            root_dir=Path(evaluation.root_dir),
            path_of_model=Path(self.config.training.trained_model_path),
            test_dir=Path(self.config.data_preparation.root_dir) / "test",
            report_path=Path(evaluation.report_path),
            scores_path=Path(evaluation.scores_path),
            serving_model_path=Path(evaluation.serving_model_path),
            all_params=self.params.to_dict(),
            # Credentials stay in the environment (MLFLOW_TRACKING_USERNAME /
            # MLFLOW_TRACKING_PASSWORD). Without a URI, runs go to ./mlruns.
            mlflow_uri=os.getenv("MLFLOW_TRACKING_URI", ""),
            experiment_name=evaluation.experiment_name,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_min_accuracy=self.params.PROMOTION_MIN_ACCURACY,
        )
