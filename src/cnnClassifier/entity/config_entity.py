from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_source: Path
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataPreparationConfig:
    root_dir: Path
    source_dir: Path
    report_path: Path
    params_val_split: float
    params_test_split: float
    params_seed: int


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    history_path: Path
    updated_base_model_path: Path
    train_dir: Path
    val_dir: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_patience: int
    params_seed: int


@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    path_of_model: Path
    test_dir: Path
    report_path: Path
    scores_path: Path
    serving_model_path: Path
    all_params: dict
    mlflow_uri: str
    experiment_name: str
    params_image_size: list
    params_batch_size: int
    params_min_accuracy: float
