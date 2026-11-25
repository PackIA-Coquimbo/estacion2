"""
Pruebas de calificación por naranja con varias fotos (promedio de predicciones).
Formato de nombres: <uid>-<n_naranja>.<n_foto>[.ext]  ej: 7712323-1.1.jpg, 7712323-1.2.jpg
"""

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (
    DEFAULT_CLASSES,
    QualityPredictor,
    aggregate_probabilities,
    classify_oranges,
    group_photos_by_orange,
    parse_orange_photo_name,
)


def make_stub_predictor(probs_by_value):
    """Predictor sin modelo: devuelve probabilidades según el valor de píxel del frame."""
    predictor = QualityPredictor.__new__(QualityPredictor)
    predictor.class_names = DEFAULT_CLASSES

    def fake_infer(frame):
        return np.asarray(probs_by_value[int(frame[0, 0, 0])], dtype=np.float32), 1.0, 2.0

    predictor._infer_probs = fake_infer
    return predictor


@pytest.mark.parametrize(
    "name, expected",
    [
        ("7712323-1.1", ("7712323", 1, 1)),
        ("7712323-1.2.jpg", ("7712323", 1, 2)),
        ("7712323-12.2.PNG", ("7712323", 12, 2)),
        ("lote-A-7712323-3.1.jpeg", ("lote-A-7712323", 3, 1)),
        ("7712323.jpg", None),
        ("7712323-1.jpg", None),
        ("foto.jpg", None),
    ],
)
def test_parse_orange_photo_name(name, expected):
    assert parse_orange_photo_name(name) == expected


def test_group_photos_by_orange_orders_and_separates():
    paths = ["7712323-2.2.jpg", "7712323-1.2.jpg", "otra.jpg", "7712323-1.1.jpg", "7712323-2.1.jpg"]
    groups, unmatched = group_photos_by_orange(paths)

    assert list(groups.keys()) == [("7712323", 1), ("7712323", 2)]
    assert [p.name for p in groups[("7712323", 1)]] == ["7712323-1.1.jpg", "7712323-1.2.jpg"]
    assert [p.name for p in groups[("7712323", 2)]] == ["7712323-2.1.jpg", "7712323-2.2.jpg"]
    assert [p.name for p in unmatched] == ["otra.jpg"]


def test_aggregate_probabilities_averages_photos():
    res = aggregate_probabilities([[0.9, 0.1, 0.0], [0.3, 0.6, 0.1]], DEFAULT_CLASSES)

    assert res["predicted_class"] == "buen_estado"
    assert res["confidence"] == pytest.approx(0.6)
    assert res["confidence_percent"] == pytest.approx(60.0)
    assert res["probabilities"] == pytest.approx(
        {"buen_estado": 0.6, "marginalmente_buena": 0.35, "mal_estado": 0.05}
    )
    assert res["photos_agree"] is False


def test_aggregate_probabilities_rejects_bad_input():
    with pytest.raises(ValueError):
        aggregate_probabilities([], DEFAULT_CLASSES)
    with pytest.raises(ValueError):
        aggregate_probabilities([[0.5, 0.5]], DEFAULT_CLASSES)


def test_predict_orange_returns_average_and_per_photo_detail():
    predictor = make_stub_predictor({10: [0.1, 0.2, 0.7], 20: [0.1, 0.4, 0.5]})
    frames = [np.full((4, 4, 3), 10, np.uint8), np.full((4, 4, 3), 20, np.uint8)]

    res = predictor.predict_orange(frames, photo_names=["x-1.1.jpg", "x-1.2.jpg"])

    assert res["predicted_class"] == "mal_estado"
    assert res["confidence_percent"] == pytest.approx(60.0)
    assert res["photos_agree"] is True
    assert res["photos_count"] == 2
    assert [p["image_name"] for p in res["photos"]] == ["x-1.1.jpg", "x-1.2.jpg"]
    assert res["photos"][0]["confidence_percent"] == pytest.approx(70.0)
    assert res["latency_ms"] == pytest.approx(6.0)


def test_classify_oranges_from_files(tmp_path):
    cv2 = pytest.importorskip("cv2")
    predictor = make_stub_predictor({
        10: [0.8, 0.1, 0.1], 20: [0.6, 0.3, 0.1],   # naranja 1 -> buen_estado
        30: [0.1, 0.2, 0.7],                         # naranja 2 (solo 1 foto)
    })
    # PNG para que el valor de píxel no cambie por compresión
    for name, value in [("7712323-1.1.png", 10), ("7712323-1.2.png", 20), ("7712323-2.1.png", 30)]:
        cv2.imwrite(str(tmp_path / name), np.full((8, 8, 3), value, np.uint8))

    results = classify_oranges(predictor, sorted(tmp_path.iterdir()), expected_photos=2)

    assert [r["orange_id"] for r in results] == ["7712323-1", "7712323-2"]
    first, second = results
    assert first["predicted_class"] == "buen_estado"
    assert first["confidence_percent"] == pytest.approx(70.0)
    assert first["complete"] is True
    assert second["photos_count"] == 1
    assert second["complete"] is False
