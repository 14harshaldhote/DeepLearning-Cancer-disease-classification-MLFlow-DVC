"""Grad-CAM: highlights the image regions that drove the model's decision."""

import numpy as np
import tensorflow as tf

LAST_CONV_LAYER = "block5_conv3"


class GradCAM:
    """Heatmap of where the last conv layer supported the predicted class.

    Gradients are taken on the logits of the final Dense layer, not the softmax
    output: a confident softmax saturates and its gradients vanish.
    """

    def __init__(self, model: tf.keras.Model, layer_name: str = LAST_CONV_LAYER):
        self.head = model.layers[-1]
        self.grad_model = tf.keras.models.Model(
            model.inputs, [model.get_layer(layer_name).output, model.layers[-2].output]
        )

    def heatmap(self, batch: np.ndarray, class_index: int) -> np.ndarray:
        """Return a heatmap in [0, 1] with the conv layer's spatial size (14x14 for VGG16)."""
        with tf.GradientTape() as tape:
            conv_out, features = self.grad_model(batch)
            logits = tf.matmul(features, self.head.kernel) + self.head.bias
            score = logits[:, class_index]
        grads = tape.gradient(score, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1)).numpy()
        return cam / cam.max() if cam.max() > 0 else cam


def jet_colormap(values: np.ndarray) -> np.ndarray:
    """Map [0, 1] values to RGB uint8 with a jet-like palette (no matplotlib needed)."""
    v = np.clip(values, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap over an RGB uint8 image of any size."""
    h, w = image_rgb.shape[:2]
    resized = tf.image.resize(heatmap[..., None], (h, w), method="bilinear").numpy()[..., 0]
    colored = jet_colormap(resized).astype(np.float32)
    blended = (1 - alpha) * image_rgb.astype(np.float32) + alpha * colored
    return blended.clip(0, 255).astype(np.uint8)
