"""
Captura asíncrona de cámara USB / V4L2 con OpenCV (Fase 3: Ingesta Asíncrona)
(Estación 2: Análisis Visual de Calidad - OrangeQuality-AI)
"""

from typing import Any, Optional, Tuple, Union
import threading
import time

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False


class FastCameraReader:
    """
    Capturador de video asíncrono con hilo en segundo plano (Fase 3: Ingesta Asíncrona).
    Lee continuamente de la cámara descartando frames viejos del buffer V4L2/USB,
    garantizando que read_latest() entregue siempre el fotograma más actual con 0 ms de retardo.
    """
    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        fourcc: str = "MJPG",
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc

        self.cap: Optional[Any] = None
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.grabbed = False
        self.frame_count = 0

    def start(self) -> "FastCameraReader":
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV no está disponible. Instale 'opencv-python' o 'opencv-python-headless'.")

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el dispositivo de cámara con índice/ruta: {self.source}")

        # Configuración de hardware USB para Raspberry Pi (DoD Fase 3)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Forzar FourCC MJPG para no saturar bus USB
        if self.fourcc and len(self.fourcc) == 4:
            fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)

        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True, name="FastCameraReaderThread")
        self.thread.start()

        # Esperar a que el primer frame esté disponible
        t_wait = time.time() + 3.0
        while time.time() < t_wait:
            with self.lock:
                if self.latest_frame is not None:
                    break
            time.sleep(0.01)

        print(f"[FastCameraReader] Cámara USB iniciada en fuente '{self.source}': {self.width}x{self.height} @ {self.fps} FPS ({self.fourcc})")
        return self

    def _worker(self) -> None:
        """Hilo dedicado a drenar continuamente el buffer de la cámara."""
        while self.running and self.cap is not None and self.cap.isOpened():
            grabbed, frame = self.cap.read()
            if not grabbed or frame is None:
                time.sleep(0.005)
                continue

            with self.lock:
                self.latest_frame = frame
                self.grabbed = True
                self.frame_count += 1

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retorna (True, frame) con el fotograma más actual en memoria,
        descartando frames atrasados acumulados en el buffer.
        """
        with self.lock:
            if not self.grabbed or self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "FastCameraReader":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

