from __future__ import annotations

import numpy as np

from prepare import _binary_auroc, _box_iou, _confusion


def test_binary_metrics_are_exact() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    assert _binary_auroc(labels, scores) == 1.0
    metrics = _confusion(labels, scores >= 0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_box_iou() -> None:
    assert _box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _box_iou((0, 0, 5, 5), (6, 6, 9, 9)) == 0.0
