import shutil
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the rank (Mann-Whitney U) formula."""
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def classification_report(y_true: np.ndarray, probs: np.ndarray, class_names: list) -> dict:
    y_pred = probs.argmax(axis=1)
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1

    per_class = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        precision = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        recall = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[i, :].sum()),
        }

    # Cancer (class 0) is the positive class for sensitivity / specificity.
    positive = (y_true == 0).astype(int)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "sensitivity": per_class[class_names[0]]["recall"],
        "specificity": per_class[class_names[1]]["recall"],
        "macro_f1": float(np.mean([c["f1"] for c in per_class.values()])),
        "roc_auc": roc_auc(positive, probs[:, 0]),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "class_names": class_names,
        "n_samples": int(len(y_true)),
    }


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _test_generator(self):
        self.test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255
        ).flow_from_directory(
            directory=self.config.test_dir,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            shuffle=False,
        )

    def evaluation(self):
        self.model = tf.keras.models.load_model(self.config.path_of_model)
        self._test_generator()
        self.loss, _ = self.model.evaluate(self.test_generator, verbose=0)
        probs = self.model.predict(self.test_generator, verbose=0)
        class_names = sorted(self.test_generator.class_indices, key=self.test_generator.class_indices.get)
        self.report = classification_report(self.test_generator.classes, probs, class_names)
        self.report["loss"] = float(self.loss)
        self.report["evaluated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        logger.info(
            f"Test accuracy {self.report['accuracy']:.3f}, AUC {self.report['roc_auc']:.3f}, "
            f"n={self.report['n_samples']}"
        )

    def save_score(self):
        save_json(
            path=self.config.scores_path,
            data={
                "loss": self.report["loss"],
                "accuracy": self.report["accuracy"],
                "sensitivity": self.report["sensitivity"],
                "specificity": self.report["specificity"],
                "macro_f1": self.report["macro_f1"],
                "roc_auc": self.report["roc_auc"],
                "n_test_images": self.report["n_samples"],
            },
        )

    def promote_model(self) -> bool:
        """Quality gate: only a model that clears the bar replaces the served one."""
        passed = self.report["accuracy"] >= self.config.params_min_accuracy
        self.report["promotion"] = {
            "min_accuracy": self.config.params_min_accuracy,
            "passed": passed,
        }
        if passed:
            self.config.serving_model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.config.path_of_model, self.config.serving_model_path)
            logger.info(f"Promoted model to {self.config.serving_model_path}")
        else:
            logger.warning("Model did not pass the quality gate; serving model unchanged")
        self.config.report_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path=Path(self.config.report_path), data=self.report)
        return passed

    def log_into_mlflow(self):
        # A remote server (e.g. DagsHub) when configured, otherwise a local SQLite store
        # (browse it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`).
        mlflow.set_tracking_uri(self.config.mlflow_uri or "sqlite:///mlflow.db")
        mlflow.set_experiment(self.config.experiment_name)

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {
                    k: self.report[k]
                    for k in ("loss", "accuracy", "sensitivity", "specificity", "macro_f1", "roc_auc")
                }
            )
            mlflow.log_artifact(str(self.config.report_path))
            mlflow.log_artifact(str(self.config.path_of_model))
        logger.info(f"Logged run to MLflow ({mlflow.get_tracking_uri()})")
