import numpy as np
import pytest

from firespread.common.metrics import average_precision, segmentation_metrics


def test_ignored_pixels_are_excluded():
    m = segmentation_metrics(np.array([1, 1, 0, 0]), np.array([1, -1, 0, 1]))
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
    assert m["fp"] == 0


def test_average_precision():
    assert average_precision([0.9, 0.8, 0.1], [1, 1, 0]) == pytest.approx(1.0)
    assert average_precision([0.9, 0.8, 0.7], [1, 0, 1]) == pytest.approx(0.5 + 0.5 * 2 / 3)
