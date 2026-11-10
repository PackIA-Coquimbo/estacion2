"""
Módulo de Soporte para Cámara Oficial Raspberry Pi (Picamera2 / CSI)
(Estación 2: Análisis Visual de Calidad - OrangeQuality-AI)

Aprovecha el procesador de señal de imagen (ISP por hardware) de la Raspberry Pi 4 / 5
utilizando Picamera2 y libcamera con salida directa a NumPy en formato RGB.
Cero consumo de CPU para demosaicing y balance de blancos.
"""

from typing import Tuple, Optional, Dict, Any
import threading
import time
import numpy as np

try:
    from picam2 import Picamera2
    HAS_PICAM2 = True
except ImportError:
    HAS_PICAM2 = False
    Picamera2 = None


def is_picam2_available() -> bool:
    """Verifica si la librería Picamera2 y el hardware CSI están disponibles."""
    if not HAS_PICAM2:
        return False
    try:
        # Intentar inicialización de prueba ligera
        p = Picamera2()
        p.close()
        return True
    except Exception:
        return False


class PiCameraReader:
    """
    Capturador asíncrono para cámara CSI oficial de Raspberry Pi (Camera Module v2 / v3 / HQ).
    Entrega frames en formato RGB directamente a arrays NumPy sin costo de conversión OpenCV.
    """
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.width = width
        self.height = height
        self.fps = fps

        self.picam2: Optional[Any] = None
        self.lock = threading.Lock()
        self.latest_frame_rgb: Optional[np.ndarray] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_count = 0
        self.grabbed = False

    def start(self) -> "PiCameraReader":
        if not HAS_PICAM2:
            raise RuntimeError(
                "Picamera2 no está disponible. Este módulo requiere Raspberry Pi OS "
                "con soporte libcamera ('pip install picamera2')."
            )

        self.picam2 = Picamera2()
        # Configurar salida directa RGB888 a la resolución objetivo
        video_config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameRate": self.fps},
        )
        self.picam2.configure(video_config)
        self.picam2.start()

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name="PiCam2Thread")
        self.thread.start()

        # Esperar primer fotograma disponible
        timeout = time.time() + 4.0
        while time.time() < timeout:
            with self.lock:
                if self.latest_frame_rgb is not None:
                    break
            time.sleep(0.01)

        print(f"[PiCameraReader] Cámara CSI iniciada: {self.width}x{self.height} a {self.fps} FPS (ISP Hardware RGB)")
        return self

    def _capture_loop(self) -> None:
        """Bucle en hilo dedicado para lectura continua a máxima velocidad sin retardo de buffer."""
        while self.running and self.picam2 is not None:
            try:
                frame_rgb = self.picam2.capture_array()
                if frame_rgb is None:
                    time.sleep(0.005)
                    continue

                with self.lock:
                    self.latest_frame_rgb = frame_rgb
                    self.grabbed = True
                    self.frame_count += 1
            except Exception as e:
                print(f"[PiCameraReader] Error en hilo de captura: {e}")
                time.sleep(0.02)

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retorna (True, frame_rgb) con el frame más reciente capturado.
        Retorna formato RGB directamente listo para preprocesamiento.
        """
        with self.lock:
            if not self.grabbed or self.latest_frame_rgb is None:
                return False, None
            return True, self.latest_frame_rgb.copy()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None
        print("[PiCameraReader] Cámara CSI detenida.")

    def __enter__(self) -> "PiCameraReader":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

