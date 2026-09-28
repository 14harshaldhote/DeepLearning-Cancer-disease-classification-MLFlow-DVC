import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        tf.keras.utils.set_random_seed(config.params_seed)

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self):
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.val_dir, shuffle=False, **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0 / 255,
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.train_dir,
            shuffle=True,
            seed=self.config.params_seed,
            **dataflow_kwargs,
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.config.params_patience,
            restore_best_weights=True,
        )

        # After de-duplication "normal" is the minority class, so weight classes
        # inversely to their frequency instead of letting the model ignore it.
        labels = self.train_generator.classes
        counts = np.bincount(labels)
        class_weight = {i: len(labels) / (len(counts) * c) for i, c in enumerate(counts)}
        logger.info(f"Class weights: {class_weight}")

        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            class_weight=class_weight,
            callbacks=[early_stopping],
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)

        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        history_dict["best_epoch"] = int(early_stopping.best_epoch) + 1
        self.config.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.history_path.write_text(json.dumps(history_dict, indent=4))
        logger.info(
            f"Trained {len(history_dict['loss'])} epochs, best epoch {history_dict['best_epoch']}"
        )
