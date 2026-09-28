import numpy as np
import pytest

from cnnClassifier.components.model_evaluation import classification_report, roc_auc


def test_roc_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_classification_report_counts():
    y_true = np.array([0, 0, 0, 1, 1])
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4]])
    report = classification_report(y_true, probs, ["adenocarcinoma", "normal"])

    assert report["confusion_matrix"] == [[2, 1], [1, 1]]
    assert report["accuracy"] == pytest.approx(0.6)
    assert report["sensitivity"] == pytest.approx(2 / 3)
    assert report["specificity"] == pytest.approx(0.5)
