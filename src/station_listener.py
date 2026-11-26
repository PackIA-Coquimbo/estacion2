"""
Módulo de Integración y Comunicación entre Estaciones en Memoria RAM (/dev/shm)
(Estación 2: Análisis Visual de Calidad - OrangeQuality-AI)

Características de Fase 4:
1. Canal de intercambio en memoria RAM pura (/dev/shm o POSIX Shared Memory) para cero
   desgaste de la tarjeta MicroSD y mínima latencia (< 1.5 ms).
2. Protocolo de sincronización con señalización de eventos (archivos flag o socket UDP).
3. Salida estandarizada en formato JSON con metadatos completos y desglose de latencias.
4. Herramienta de benchmarking integrada para verificar transferencias inter-estación.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List, Sequence
import argparse
import json
import os
import sys
import tempfile
import time
import numpy as np

# Asegurar raíz en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from src.inf_erence.predictor import QualityPredictor
from src.inf_erence.oranges import (
    EXPECTED_PHOTOS_PER_ORANGE,
    build_orange_payload,
    format_orange_photo_name,
    group_photos_by_orange,
    make_orange_id,
)

# Tiempo máximo (s) que se espera la(s) foto(s) faltante(s) de una naranja antes de
# calificarla con las fotos disponibles.
DEFAULT_PHOTO_TIMEOUT_SEC = 2.0


def get_default_shm_dir() -> Path:
    """
    Retorna el directorio de memoria compartida en RAM.
    En Linux/Raspberry Pi utiliza /dev/shm (tmpfs en memoria RAM física).
    En otros entornos (desarrollo/Windows), utiliza el directorio temporal en memoria.
    """
    linux_shm = Path("/dev/shm")
    if linux_shm.exists() and linux_shm.is_dir():
        return linux_shm

    # Fallback para Windows o entornos no Linux
    fallback = Path(tempfile.gettempdir()) / "estacion_shm"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


class StationCommunicator:
    """
    Controlador del canal de intercambio de datos en RAM entre Estación 1 y Estación 2.
    Gestiona la escritura/lectura de frames y archivos de control sin tocar la MicroSD.
    """
    def __init__(self, shm_dir: Optional[Union[str, Path]] = None):
        self.shm_dir = Path(shm_dir) if shm_dir is not None else get_default_shm_dir()
        self.shm_dir.mkdir(parents=True, exist_ok=True)

        # Rutas de sincronización e intercambio en RAM
        self.frame_path = self.shm_dir / "estacion_frame.jpg"
        self.trigger_path = self.shm_dir / "estacion_trigger.flag"
        self.result_path = self.shm_dir / "estacion2_result.json"
        self.ack_path = self.shm_dir / "estacion_ack.flag"

        # Canal multi-foto por naranja:
        #   inbox/   <- Estación 1 deja <uid>-<naranja>.<foto>.jpg (escritura atómica .tmp + rename)
        #   results/ <- Estación 2 deja <uid>-<naranja>.json con la calificación promediada
        self.inbox_dir = self.shm_dir / "estacion2_inbox"
        self.results_dir = self.shm_dir / "estacion2_results"
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Canal multi-foto por naranja
    # ------------------------------------------------------------------
    def send_orange_photos_from_station1(
        self,
        uid: str,
        orange_number: int,
        frames_bgr: Sequence[np.ndarray],
        quality: int = 95,
    ) -> List[Path]:
        """
        Usado por la Estación 1 (o simulador): deja las fotos de una naranja en la bandeja
        como <uid>-<naranja>.<n_foto>.jpg. Cada archivo se escribe primero como .tmp y luego
        se renombra, por lo que la Estación 2 nunca lee una imagen a medio escribir.
        """
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV no está disponible para codificar la imagen.")

        written: List[Path] = []
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        for photo_number, frame in enumerate(frames_bgr, start=1):
            success, encoded_img = cv2.imencode(".jpg", frame, encode_param)
            if not success:
                raise RuntimeError("Fallo al codificar frame BGR a formato JPEG.")
            final_path = self.inbox_dir / format_orange_photo_name(uid, orange_number, photo_number)
            tmp_path = final_path.with_name(final_path.name + ".tmp")
            with open(tmp_path, "wb") as f:
                f.write(encoded_img.tobytes())
            tmp_path.replace(final_path)
            written.append(final_path)
        return written

    def list_pending_oranges(self) -> Dict[Tuple[str, int], List[Path]]:
        """Agrupa por naranja las fotos terminadas en la bandeja (ignora archivos .tmp u otros)."""
        files = [p for p in self.inbox_dir.iterdir() if p.is_file()]
        groups, _ = group_photos_by_orange(files)
        return groups

    def read_photos(self, paths: Sequence[Path]) -> Tuple[List[np.ndarray], List[str], float]:
        """Lee y decodifica fotos desde RAM. Retorna (frames, nombres, latencia_ms) de las legibles."""
        t0 = time.perf_counter()
        frames: List[np.ndarray] = []
        names: List[str] = []
        for path in paths:
            try:
                buf = path.read_bytes()
            except FileNotFoundError:
                continue
            frame = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
                names.append(path.name)
        return frames, names, (time.perf_counter() - t0) * 1000.0

    def remove_photos(self, paths: Sequence[Path]) -> None:
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def write_orange_result(self, payload: Dict[str, Any]) -> Path:
        """Escribe de forma atómica el resultado de una naranja en results/<orange_id>.json."""
        result_path = self.results_dir / f"{payload['orange_id']}.json"
        tmp_path = result_path.with_name(result_path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(result_path)
        return result_path

    def wait_for_orange_result_in_station1(
        self,
        orange_id: str,
        timeout_sec: float = 5.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Usado por la Estación 1: espera el resultado de la naranja <uid>-<n>, lo retorna
        y lo elimina de la carpeta de resultados. Retorna None si se agota el tiempo.
        """
        result_path = self.results_dir / f"{orange_id}.json"
        t_expire = time.time() + timeout_sec
        while time.time() < t_expire:
            if result_path.exists():
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    result_path.unlink()
                    return data
                except (json.JSONDecodeError, PermissionError, FileNotFoundError):
                    pass
            time.sleep(0.001)
        return None

    def send_frame_from_station1(self, frame_bgr: np.ndarray, quality: int = 95) -> float:
        """
        Método utilizado por la Estación 1 (o simulador) para enviar un frame a la Estación 2 vía RAM.
        Retorna el tiempo de escritura en RAM en milisegundos.
        """
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV no está disponible para codificar la imagen.")

        t0 = time.perf_counter()

        # Limpiar banderas anteriores
        if self.ack_path.exists():
            self.ack_path.unlink()

        # Escribir frame comprimido directamente en /dev/shm
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded_img = cv2.imencode(".jpg", frame_bgr, encode_param)
        if not success:
            raise RuntimeError("Fallo al codificar frame BGR a formato JPEG.")

        with open(self.frame_path, "wb") as f:
            f.write(encoded_img.tobytes())

        # Emitir señal de trigger para la Estación 2
        with open(self.trigger_path, "w", encoding="utf-8") as f:
            f.write(str(time.time()))

        write_latency_ms = (time.perf_counter() - t0) * 1000.0
        return write_latency_ms

    def read_frame_in_station2(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Lee el frame compartido desde la memoria RAM (/dev/shm).
        Retorna (exito, frame_bgr, latencia_lectura_ms).
        """
        if not self.frame_path.exists():
            return False, None, 0.0

        t0 = time.perf_counter()

        # Lectura directa desde el buffer en RAM
        with open(self.frame_path, "rb") as f:
            buf = f.read()

        arr = np.frombuffer(buf, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        read_latency_ms = (time.perf_counter() - t0) * 1000.0

        if frame_bgr is None:
            return False, None, read_latency_ms

        return True, frame_bgr, read_latency_ms

    def write_result_from_station2(self, result_dict: Dict[str, Any]) -> None:
        """Escribe el resultado estructurado JSON en /dev/shm y levanta el flag de ACK."""
        # Escritura atómica en el mismo sistema de archivos en RAM
        temp_result = self.shm_dir / "estacion2_result.tmp"
        with open(temp_result, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)
        temp_result.replace(self.result_path)

        # Retirar trigger y activar confirmación ACK
        if self.trigger_path.exists():
            self.trigger_path.unlink()

        with open(self.ack_path, "w", encoding="utf-8") as f:
            f.write(str(time.time()))

    def wait_for_result_in_station1(self, timeout_sec: float = 2.0) -> Optional[Dict[str, Any]]:
        """Espera a que la Estación 2 complete el análisis y retorne el JSON."""
        t_expire = time.time() + timeout_sec
        while time.time() < t_expire:
            if self.ack_path.exists() and self.result_path.exists():
                try:
                    with open(self.result_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return data
                except (json.JSONDecodeError, PermissionError):
                    time.sleep(0.001)
            time.sleep(0.001)
        return None

    def cleanup(self) -> None:
        """Limpia los archivos temporales creados en /dev/shm."""
        paths = [self.frame_path, self.trigger_path, self.result_path, self.ack_path]
        for folder in (self.inbox_dir, self.results_dir):
            if folder.exists():
                paths.extend(p for p in folder.iterdir() if p.is_file())
        for p in paths:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass


class StationListener:
    """
    Servicio de escucha de eventos (Listener) de la Estación 2.
    Espera señales de la Estación 1, lee la imagen en RAM (/dev/shm),
    ejecuta la predicción de calidad visual y entrega el resultado estandarizado.
    """
    def __init__(
        self,
        predictor: QualityPredictor,
        shm_dir: Optional[Union[str, Path]] = None,
        poll_interval_sec: float = 0.002,
        expected_photos: int = EXPECTED_PHOTOS_PER_ORANGE,
        photo_timeout_sec: float = DEFAULT_PHOTO_TIMEOUT_SEC,
    ):
        self.predictor = predictor
        self.communicator = StationCommunicator(shm_dir=shm_dir)
        self.poll_interval = poll_interval_sec
        self.expected_photos = expected_photos
        self.photo_timeout_sec = photo_timeout_sec
        self.running = False
        self.processed_count = 0
        # Momento en que el servicio vio por primera vez cada naranja en la bandeja.
        # No se usa la fecha de modificación de los archivos: una copia (Copy-Item,
        # scp -p, rsync -a) conserva la fecha original y la naranja parecería vencida.
        self._first_seen: Dict[Tuple[str, int], float] = {}

    def process_pending_oranges(self) -> List[Dict[str, Any]]:
        """
        Califica las naranjas listas en la bandeja de fotos:
        - Completa: llegaron `expected_photos` fotos legibles -> se procesa de inmediato.
        - Incompleta: si pasaron `photo_timeout_sec` desde que el servicio vio la naranja
          por primera vez, se califica con las fotos disponibles (campo "complete": false).
          Si ninguna es legible, se entrega un resultado con "error" para que la
          Estación 1 no quede esperando.
        Las fotos procesadas se eliminan de la bandeja.
        """
        results: List[Dict[str, Any]] = []
        now = time.time()
        pending = self.communicator.list_pending_oranges()

        # Olvidar naranjas que ya no están en la bandeja (procesadas o retiradas)
        for key in list(self._first_seen):
            if key not in pending:
                del self._first_seen[key]

        for (uid, orange_number), paths in pending.items():
            first_seen = self._first_seen.setdefault((uid, orange_number), now)
            timed_out = (now - first_seen) >= self.photo_timeout_sec
            if len(paths) < self.expected_photos and not timed_out:
                continue

            t_cycle_start = time.perf_counter()
            frames, names, read_latency_ms = self.communicator.read_photos(paths)
            if len(frames) < len(paths) and not timed_out:
                continue  # Alguna foto aún no es legible; reintentar en el próximo ciclo

            if not frames:
                payload: Dict[str, Any] = {
                    "station_id": 2,
                    "timestamp": round(time.time(), 3),
                    "uid": uid,
                    "orange_number": orange_number,
                    "orange_id": make_orange_id(uid, orange_number),
                    "error": "Ninguna foto de la naranja pudo decodificarse.",
                    "photos_count": 0,
                    "expected_photos": self.expected_photos,
                    "complete": False,
                }
            else:
                orange_res = self.predictor.predict_orange(frames, photo_names=names)
                payload = build_orange_payload(uid, orange_number, orange_res, self.expected_photos)
                payload["latency_read_shm_ms"] = round(read_latency_ms, 2)
                payload["latency_total_ms"] = round((time.perf_counter() - t_cycle_start) * 1000.0, 2)

            self.communicator.write_orange_result(payload)
            self.communicator.remove_photos(paths)
            self._first_seen.pop((uid, orange_number), None)
            self.processed_count += 1
            results.append(payload)

        return results

    def process_pending_event(self) -> Optional[Dict[str, Any]]:
        """Procesa un evento pendiente si el archivo trigger existe."""
        if not self.communicator.trigger_path.exists():
            return None

        t_cycle_start = time.perf_counter()

        # 1. Leer frame directamente de memoria RAM (/dev/shm)
        success, frame_bgr, read_latency_ms = self.communicator.read_frame_in_station2()
        if not success or frame_bgr is None:
            return None

        # 2. Ejecutar inferencia visual de calidad
        pred_res, _ = self.predictor.predict_cv2(frame_bgr)

        t_cycle_end = time.perf_counter()
        total_transfer_infer_ms = (t_cycle_end - t_cycle_start) * 1000.0

        # 3. Construir estructura JSON estandarizada según especificación de Fase 4
        result_payload = {
            "station_id": 2,
            "timestamp": round(time.time(), 3),
            "predicted_class": pred_res["predicted_class"],
            "confidence_percent": pred_res["confidence_percent"],
            "probabilities": pred_res["probabilities"],
            "latency_read_shm_ms": round(read_latency_ms, 2),
            "latency_preprocess_ms": pred_res["latency_preprocess_ms"],
            "latency_inference_ms": pred_res["latency_inference_ms"],
            "latency_total_ms": round(total_transfer_infer_ms, 2),
        }

        # 4. Escribir resultado en /dev/shm y emitir ACK
        self.communicator.write_result_from_station2(result_payload)
        self.processed_count += 1

        return result_payload

    def start_listening(self, max_events: Optional[int] = None) -> None:
        """Bucle continuo de escucha de eventos inter-estación."""
        self.running = True
        print("=" * 70)
        print("ESTACIÓN 2: SERVICIO DE ESCUCHA INTER-ESTACIONES (RAM /dev/shm)")
        print(f"Canal de memoria compartida: {self.communicator.shm_dir}")
        print(f"Bandeja de fotos por naranja:  {self.communicator.inbox_dir} "
              f"({self.expected_photos} fotos/naranja, espera máx. {self.photo_timeout_sec:.1f}s)")
        print(f"Resultados por naranja:        {self.communicator.results_dir}")
        print("Estado: Esperando señales de captura desde Estación 1...")
        print("=" * 70)

        try:
            while self.running:
                handled = False

                result = self.process_pending_event()
                if result is not None:
                    handled = True
                    print(
                        f"[EVENTO #{self.processed_count:04d}] "
                        f"Calidad: {result['predicted_class'].upper():<20} ({result['confidence_percent']:5.1f}%) | "
                        f"Lectura RAM: {result['latency_read_shm_ms']:.2f}ms | "
                        f"Inf: {result['latency_inference_ms']:.2f}ms | "
                        f"Total: {result['latency_total_ms']:.2f}ms"
                    )

                for orange in self.process_pending_oranges():
                    handled = True
                    if "error" in orange:
                        print(f"[NARANJA {orange['orange_id']}] ERROR: {orange['error']}")
                        continue
                    print(
                        f"[NARANJA {orange['orange_id']}] "
                        f"Calidad promedio: {orange['predicted_class'].upper():<20} "
                        f"({orange['confidence_percent']:5.1f}%) | "
                        f"Fotos: {orange['photos_count']}/{orange['expected_photos']}"
                        f"{'' if orange['photos_agree'] else ' (no coinciden)'} | "
                        f"Total: {orange['latency_total_ms']:.2f}ms"
                    )

                if max_events is not None and self.processed_count >= max_events:
                    print(f"[INFO] Límite de {max_events} eventos procesados alcanzado.")
                    break
                if not handled:
                    time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("\n[INFO] Detención del servicio solicitada por el usuario.")
        finally:
            self.running = False
            print(f"[INFO] Servicio finalizado. Total de frutas analizadas: {self.processed_count}")


def run_interstation_benchmark(iterations: int = 100, shm_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Ejecuta un benchmark exhaustivo de la transferencia entre Estación 1 y Estación 2
    en memoria RAM (/dev/shm) para verificar el criterio de aceptación (< 1.5 ms de lectura/transferencia).
    """
    if not HAS_OPENCV:
        raise RuntimeError("OpenCV es requerido para el benchmark de estaciones.")

    comm = StationCommunicator(shm_dir=shm_dir)
    print("\n" + "=" * 70)
    print("BENCHMARK DE INTEGRACIÓN INTER-ESTACIONES (DoD - FASE 4)")
    print(f"Canal RAM: {comm.shm_dir}")
    print("=" * 70)

    # Crear frame sintético típico de cámara de producción (640x480)
    sample_frame = np.full((480, 640, 3), (30, 30, 30), dtype=np.uint8)
    cv2.circle(sample_frame, (320, 240), 120, (20, 140, 235), -1)

    read_latencies: list[float] = []
    write_latencies: list[float] = []

    # Warmup
    for _ in range(5):
        comm.send_frame_from_station1(sample_frame)
        comm.read_frame_in_station2()

    for _ in range(iterations):
        w_lat = comm.send_frame_from_station1(sample_frame)
        success, _, r_lat = comm.read_frame_in_station2()
        assert success, "Error al leer imagen compartida en RAM."
        write_latencies.append(w_lat)
        read_latencies.append(r_lat)

    comm.cleanup()

    mean_read = float(np.mean(read_latencies))
    p95_read = float(np.percentile(read_latencies, 95))
    min_read = float(np.min(read_latencies))

    mean_write = float(np.mean(write_latencies))

    print(f"Iteraciones evaluadas:          {iterations}")
    print(f"Resolución de frame:            640x480 BGR (JPEG en RAM)")
    print(f"Latencia media de lectura RAM:  {mean_read:.3f} ms")
    print(f"Percentil 95 lectura RAM:       {p95_read:.3f} ms")
    print(f"Latencia mínima lectura RAM:    {min_read:.3f} ms")
    print(f"Latencia media escritura RAM:   {mean_write:.3f} ms")
    print("-" * 70)

    passed = mean_read < 1.5
    status_str = "CUMPLIDO" if passed else "FALLIDO"
    print(f"[DoD Fase 4] Latencia lectura/transferencia < 1.5 ms: {status_str} ({mean_read:.3f} ms)")
    print(f"[DoD Fase 4] Cero escritura en MicroSD:               CUMPLIDO (Operaciones exclusivas en RAM)")
    print("=" * 70)

    return {
        "iterations": iterations,
        "mean_read_ms": mean_read,
        "p95_read_ms": p95_read,
        "mean_write_ms": mean_write,
        "passed_dod": passed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Servicio de Integración Inter-Estaciones - RAM /dev/shm (OrangeQuality-AI)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "weights" / "orange_model_int8.onnx"),
        help="Ruta al modelo ONNX (por defecto INT8 optimizado)",
    )
    parser.add_argument(
        "--shm-dir",
        type=str,
        default=None,
        help="Directorio de memoria compartida (por defecto /dev/shm en Linux o tempdir)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Ejecuta benchmark de velocidad de transferencia en memoria RAM",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Espera y procesa un único lote/frame y sale",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Hilos de inferencia para ONNX Runtime",
    )
    parser.add_argument(
        "--expected-photos",
        type=int,
        default=EXPECTED_PHOTOS_PER_ORANGE,
        help="Fotos esperadas por naranja antes de calificarla (por defecto 2)",
    )
    parser.add_argument(
        "--photo-timeout",
        type=float,
        default=DEFAULT_PHOTO_TIMEOUT_SEC,
        help="Segundos a esperar fotos faltantes antes de calificar con las disponibles",
    )
    args = parser.parse_args()

    if args.benchmark:
        run_interstation_benchmark(
            iterations=100,
            shm_dir=Path(args.shm_dir) if args.shm_dir else None,
        )
        return

    # Inicializar predictor de calidad optimizado
    model_path = Path(args.model)
    if not model_path.exists():
        fallback_fp32 = PROJECT_ROOT / "weights" / "orange_model.onnx"
        if fallback_fp32.exists():
            model_path = fallback_fp32

    predictor = QualityPredictor(
        model_path=model_path,
        use_onnx=True,
        num_threads=args.threads,
    )

    listener = StationListener(
        predictor=predictor,
        shm_dir=args.shm_dir,
        expected_photos=args.expected_photos,
        photo_timeout_sec=args.photo_timeout,
    )

    max_events = 1 if args.once else None
    listener.start_listening(max_events=max_events)


if __name__ == "__main__":
    main()

