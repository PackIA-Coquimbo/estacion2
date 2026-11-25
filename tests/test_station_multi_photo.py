"""
Pruebas del canal inter-estaciones multi-foto (/dev/shm):
la Estación 1 deja <uid>-<naranja>.<foto>.jpg en la bandeja y la Estación 2
califica cada naranja promediando sus fotos.
"""

from pathlib import Path
import os
import sys
import time

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

cv2 = pytest.importorskip("cv2")

from src.inference import DEFAULT_CLASSES, QualityPredictor
from src.station_listener import StationListener

DARK = np.full((48, 64, 3), 40, np.uint8)
BRIGHT = np.full((48, 64, 3), 210, np.uint8)


def make_stub_predictor():
    """Predictor sin modelo: fotos oscuras -> mal_estado, claras -> buen_estado."""
    predictor = QualityPredictor.__new__(QualityPredictor)
    predictor.class_names = DEFAULT_CLASSES

    def fake_infer(frame):
        probs = [0.1, 0.2, 0.7] if frame.mean() < 128 else [0.7, 0.2, 0.1]
        return np.asarray(probs, dtype=np.float32), 1.0, 2.0

    predictor._infer_probs = fake_infer
    return predictor


def make_listener(tmp_path, timeout=5.0):
    return StationListener(
        predictor=make_stub_predictor(),
        shm_dir=tmp_path,
        expected_photos=2,
        photo_timeout_sec=timeout,
    )


def age_files(folder: Path, seconds: float) -> None:
    old = time.time() - seconds
    for p in folder.iterdir():
        os.utime(p, (old, old))


def test_complete_orange_is_averaged_and_delivered(tmp_path):
    listener = make_listener(tmp_path)
    comm = listener.communicator
    comm.send_orange_photos_from_station1("7712323", 1, [BRIGHT, DARK])

    results = listener.process_pending_oranges()

    assert len(results) == 1
    res = results[0]
    assert res["orange_id"] == "7712323-1"
    assert res["photos_count"] == 2 and res["complete"] is True
    assert res["photos_agree"] is False
    assert res["probabilities"] == pytest.approx(
        {"buen_estado": 0.4, "marginalmente_buena": 0.2, "mal_estado": 0.4}
    )
    assert [p["image_name"] for p in res["photos"]] == ["7712323-1.1.jpg", "7712323-1.2.jpg"]
    assert list(comm.inbox_dir.iterdir()) == []

    delivered = comm.wait_for_orange_result_in_station1("7712323-1", timeout_sec=0.5)
    assert delivered == res
    assert list(comm.results_dir.iterdir()) == []


def test_incomplete_orange_waits_then_is_graded_after_timeout(tmp_path):
    listener = make_listener(tmp_path, timeout=5.0)
    comm = listener.communicator
    comm.send_orange_photos_from_station1("7712323", 2, [DARK])

    assert listener.process_pending_oranges() == []
    assert len(list(comm.inbox_dir.iterdir())) == 1

    age_files(comm.inbox_dir, 10.0)
    results = listener.process_pending_oranges()

    assert len(results) == 1
    assert results[0]["predicted_class"] == "mal_estado"
    assert results[0]["photos_count"] == 1
    assert results[0]["complete"] is False


def test_multiple_oranges_and_tmp_files(tmp_path):
    listener = make_listener(tmp_path)
    comm = listener.communicator
    comm.send_orange_photos_from_station1("7712323", 1, [BRIGHT, BRIGHT])
    comm.send_orange_photos_from_station1("7712323", 2, [DARK, DARK])
    # Foto en plena escritura: no debe contarse
    (comm.inbox_dir / "7712323-3.1.jpg.tmp").write_bytes(b"parcial")

    results = listener.process_pending_oranges()

    assert {r["orange_id"]: r["predicted_class"] for r in results} == {
        "7712323-1": "buen_estado",
        "7712323-2": "mal_estado",
    }
    assert all(r["photos_agree"] for r in results)
    assert [p.name for p in comm.inbox_dir.iterdir()] == ["7712323-3.1.jpg.tmp"]


def test_unreadable_photos_produce_error_result_after_timeout(tmp_path):
    listener = make_listener(tmp_path, timeout=5.0)
    comm = listener.communicator
    (comm.inbox_dir / "7712323-4.1.jpg").write_bytes(b"no es jpeg")
    (comm.inbox_dir / "7712323-4.2.jpg").write_bytes(b"tampoco")

    assert listener.process_pending_oranges() == []  # se reintenta antes del timeout

    age_files(comm.inbox_dir, 10.0)
    results = listener.process_pending_oranges()

    assert len(results) == 1 and "error" in results[0]
    assert (comm.results_dir / "7712323-4.json").exists()
    assert list(comm.inbox_dir.iterdir()) == []


def test_start_listening_once_processes_one_orange(tmp_path):
    listener = make_listener(tmp_path)
    listener.communicator.send_orange_photos_from_station1("abc", 7, [BRIGHT, BRIGHT])

    listener.start_listening(max_events=1)

    assert listener.processed_count == 1
    assert (listener.communicator.results_dir / "abc-7.json").exists()
