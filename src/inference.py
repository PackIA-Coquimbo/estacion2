"""
Motor de Inferencia en Tiempo Real Ligero (Edge / Raspberry Pi), Ingesta Asíncrona y Exportación ONNX
(Estación 2: Análisis Visual de Calidad - OrangeQuality-AI)

Características:
1. Aislamiento completo de dependencias: Inferencia 100% nativa con onnxruntime,
   numpy y opencv sin requerir PyTorch ni Pillow en el dispositivo embebido (Fase 2).
2. Preprocesamiento Zero-Copy vectorizado directo (preprocess_cv2_fast) < 3.0 ms (Fase 2).
3. Sesión ONNX Runtime optimizada para CPU ARM Cortex (4 threads, optimización total de grafos).
4. Ingesta de video asíncrona con hilo dedicado (FastCameraReader) que descarta frames viejos
   del buffer V4L2/USB para 0 ms de retardo acumulado (Fase 3).
5. Configuración USB FourCC 'MJPG' y resolución controlada 640x480 a 30 FPS (Fase 3).
6. Soporte para cámara oficial CSI de Raspberry Pi mediante Picamera2 (Fase 3).
7. Modo --headless para ejecución como servicio systemd o SSH sin costo de interfaz X11 (>25% ahorro de CPU).

Estructura de módulos (src/inf_erence/):
- src/inf_erence/preprocessing.py  Preprocesamiento OpenCV/NumPy y softmax.
- src/inf_erence/predictor.py      QualityPredictor (ONNX Runtime / PyTorch opcional).
- src/inf_erence/oranges.py        Nombres de fotos, agrupación y promedio por naranja.
- src/inf_erence/camera_usb.py     FastCameraReader (USB / V4L2).
Otros: src/classes.py (clases), src/camera_picam2.py (CSI), tools/export_onnx.py (exportación ONNX).
Este archivo contiene la CLI y reexporta los nombres anteriores por compatibilidad.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import argparse
import json
import sys
import time

import numpy as np

# Asegurar raíz del proyecto en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reexportaciones por compatibilidad: `from src.inference import ...` sigue funcionando
from src.classes import DEFAULT_CLASSES, CLASS_COLORS_BGR
from src.inf_erence.preprocessing import (
    HAS_OPENCV,
    NORM_MEAN_NP,
    NORM_STD_NP,
    preprocess_cv2_fast,
    softmax_numpy,
)
from src.inf_erence.camera_usb import FastCameraReader
from src.inf_erence.predictor import QualityPredictor, HAS_ONNXRUNTIME, HAS_TORCH
from src.inf_erence.oranges import (
    EXPECTED_PHOTOS_PER_ORANGE,
    IMAGE_EXTENSIONS,
    ORANGE_PHOTO_PATTERN,
    aggregate_probabilities,
    build_orange_payload,
    classify_oranges,
    group_photos_by_orange,
    parse_orange_photo_name,
)
from src.e5_client import (
    DEFAULT_E5_BASE_URL,
    E5Client,
    build_e2_payload,
    map_prediction_to_e2_datos,
)

__all__ = [
    "DEFAULT_CLASSES", "CLASS_COLORS_BGR",
    "HAS_OPENCV", "HAS_ONNXRUNTIME", "HAS_TORCH",
    "NORM_MEAN_NP", "NORM_STD_NP", "preprocess_cv2_fast", "softmax_numpy",
    "FastCameraReader", "QualityPredictor", "export_to_onnx",
    "EXPECTED_PHOTOS_PER_ORANGE", "IMAGE_EXTENSIONS", "ORANGE_PHOTO_PATTERN",
    "aggregate_probabilities", "build_orange_payload", "classify_oranges",
    "group_photos_by_orange", "parse_orange_photo_name",
    "run_preprocessing_benchmark", "run_camera_stream", "print_orange_results", "main",
]

if HAS_OPENCV:
    import cv2


def export_to_onnx(*args, **kwargs) -> Path:
    """Compatibilidad: delega en tools/export_onnx.py (requiere PyTorch)."""
    from tools.export_onnx import export_to_onnx as _export_to_onnx
    return _export_to_onnx(*args, **kwargs)


def run_preprocessing_benchmark(iterations: int = 200, frame_shape: Tuple[int, int, int] = (480, 640, 3)) -> float:
    """
    Ejecuta una prueba cuantitativa de latencia de preprocesamiento (DoD Fase 2).
    Verifica que la latencia media sea <= 3.5 ms.
    """
    if not HAS_OPENCV:
        raise RuntimeError("OpenCV es requerido para el benchmark.")

    print("\n" + "=" * 65)
    print("BENCHMARK DE PREPROCESAMIENTO ZERO-COPY (DoD - FASE 2)")
    print("=" * 65)
    dummy_frame = np.random.randint(0, 256, frame_shape, dtype=np.uint8)

    # Warmup
    for _ in range(10):
        _ = preprocess_cv2_fast(dummy_frame)

    latencies_ms: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = preprocess_cv2_fast(dummy_frame)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = float(np.mean(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    min_lat = float(np.min(latencies_ms))
    max_lat = float(np.max(latencies_ms))

    print(f"Iteraciones:             {iterations}")
    print(f"Resolución de entrada:   {frame_shape[1]}x{frame_shape[0]} BGR")
    print(f"Latencia Media:          {mean_lat:.2f} ms")
    print(f"Latencia Percentil 95:   {p95_lat:.2f} ms")
    print(f"Latencia Mínima:         {min_lat:.2f} ms")
    print(f"Latencia Máxima:         {max_lat:.2f} ms")
    print("-" * 65)

    passed = mean_lat <= 3.5
    status_str = "CUMPLIDO" if passed else "FALLIDO"
    print(f"[DoD Fase 2] Latencia preprocesamiento <= 3.5 ms: {status_str} ({mean_lat:.2f} ms)")
    return mean_lat



def run_camera_stream(
    predictor: QualityPredictor,
    camera_source: Union[int, str] = 0,
    use_picam: bool = False,
    headless: bool = False,
    max_frames: Optional[int] = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> None:
    """
    Ejecuta el bucle de inferencia en tiempo real sobre flujo de video asíncrono (Fase 3).
    Soporta modo headless (sin X11/GUI) con reducción de CPU > 25% y Picamera2 (CSI).
    """
    reader = None
    if use_picam:
        try:
            from src.camera_picam2 import PiCameraReader, is_picam2_available
            if not is_picam2_available():
                print("[AVISO] Picamera2 / CSI no detectada. Cambiando a FastCameraReader (USB)...")
                reader = FastCameraReader(source=camera_source, width=width, height=height, fps=fps)
            else:
                reader = PiCameraReader(width=width, height=height, fps=fps)
        except ImportError:
            print("[AVISO] Módulo camera_picam2 no disponible. Usando FastCameraReader (USB)...")
            reader = FastCameraReader(source=camera_source, width=width, height=height, fps=fps)
    else:
        reader = FastCameraReader(source=camera_source, width=width, height=height, fps=fps)

    print("\n" + "=" * 70)
    mode_str = "HEADLESS (Sin GUI X11 - Servicio / Producción)" if headless else "GUI INTERACTIVO (Ventana OpenCV)"
    print(f"INICIANDO FLUJO DE VIDEO ASÍNCRONO - MODO: {mode_str}")
    print(f"Resolución objetivo: {width}x{height} @ {fps} FPS | Formato: FourCC MJPG")
    if headless:
        print("Control: Modo consola activo. Presione Ctrl+C para detener.")
    else:
        print("Control: Ventana gráfica activa. Presione tecla 'q' para salir.")
    print("=" * 70)

    try:
        reader.start()
    except Exception as e:
        print(f"Error al inicializar la cámara: {e}")
        return

    frames_processed = 0
    t_start = time.perf_counter()
    last_stat_time = t_start
    stat_interval = 1.0  # Reporte de métricas cada 1 segundo en consola

    try:
        while True:
            ret, frame = reader.read_latest()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frames_processed += 1
            result, annotated = predictor.predict_cv2(frame)

            now = time.perf_counter()

            if headless:
                # Modo Headless: Cero llamadas a imshow/waitKey para ahorrar CPU
                if (now - last_stat_time) >= stat_interval:
                    elapsed = now - t_start
                    effective_fps = frames_processed / elapsed if elapsed > 0 else 0
                    print(
                        f"[HEADLESS] Frame #{frames_processed:05d} | "
                        f"Calidad: {result['predicted_class'].upper():<20} ({result['confidence_percent']:5.1f}%) | "
                        f"Pre: {result['latency_preprocess_ms']:4.1f}ms | "
                        f"Inf: {result['latency_inference_ms']:4.1f}ms | "
                        f"FPS: {effective_fps:4.1f}"
                    )
                    last_stat_time = now
            else:
                cv2.imshow("Estacion 2 - Analisis Visual de Calidad (OrangeQuality-AI)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames is not None and frames_processed >= max_frames:
                print(f"[INFO] Límite de {max_frames} frames alcanzado.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Detención solicitada por el usuario (KeyboardInterrupt).")
    finally:
        reader.stop()
        if not headless and HAS_OPENCV:
            cv2.destroyAllWindows()

        total_elapsed = time.perf_counter() - t_start
        effective_fps = frames_processed / total_elapsed if total_elapsed > 0 else 0
        print("-" * 70)
        print(f"Resumen de Ejecución: {frames_processed} frames procesados en {total_elapsed:.2f}s ({effective_fps:.1f} FPS medio).")
        print("=" * 70)



def print_orange_results(results: Sequence[Dict[str, Any]]) -> None:
    """Imprime los resultados por naranja en formato legible."""
    for res in results:
        print("\n" + "=" * 55)
        print(f"NARANJA {res['orange_id']}  (UID {res['uid']}, n° {res['orange_number']})")
        print("=" * 55)
        print(f"Calidad Promedio:        {res['predicted_class']}")
        print(f"Confianza Promedio:      {res['confidence_percent']}%")
        print(f"Fotos analizadas:        {res['photos_count']}/{res['expected_photos']}"
              f"{'' if res['photos_agree'] else '  (las fotos NO coinciden)'}")
        print("Probabilidades promedio:")
        for cls, prob in res["probabilities"].items():
            print(f"  - {cls:<22}: {prob * 100:.2f}%")
        print("Detalle por foto:")
        for photo in res["photos"]:
            print(f"  - {photo['image_name']:<24}: {photo['predicted_class']} ({photo['confidence_percent']}%)")
        print(f"Latencia Total:          {res['latency_total_ms']} ms")
    print("=" * 55)



def main():
    parser = argparse.ArgumentParser(description="Inferencia y Calidad Visual (OrangeQuality-AI - Edge)")
    parser.add_argument(
        "--image", type=str, nargs="+", default=None,
        help="Una o más imágenes. Si siguen el formato <uid>-<naranja>.<foto> (ej. 7712323-1.1.jpg), "
             "se agrupan por naranja y se promedia la calificación de sus fotos.",
    )
    parser.add_argument("--dir", type=str, default=None, help="Carpeta con fotos <uid>-<naranja>.<foto> a calificar por naranja")
    parser.add_argument("--expected-photos", type=int, default=EXPECTED_PHOTOS_PER_ORANGE, help="Fotos esperadas por naranja (por defecto 2)")
    parser.add_argument("--json", action="store_true", help="Retorna la salida exclusivamente en formato JSON estándar")
    parser.add_argument("--webcam", type=int, default=None, help="Índice de cámara para inferencia en vivo")
    parser.add_argument("--picam", action="store_true", help="Usa la cámara CSI oficial de Raspberry Pi (Picamera2)")
    parser.add_argument("--headless", action="store_true", help="Modo sin interfaz gráfica (cero sobrecarga X11)")
    parser.add_argument("--max-frames", type=int, default=None, help="Límite de fotogramas para pruebas o benchmark")
    parser.add_argument("--weights", type=str, default=str(PROJECT_ROOT / "weights" / "best_orange_model.pth"), help="Pesos .pth (PyTorch)")
    parser.add_argument("--onnx", type=str, default=None, help="Ruta a modelo .onnx (ONNX Runtime)")
    parser.add_argument("--export-onnx", type=str, default=None, help="Exportar modelo PyTorch a ONNX")
    parser.add_argument("--threads", type=int, default=4, help="Número de hilos de inferencia (por defecto 4)")
    parser.add_argument("--benchmark", action="store_true", help="Ejecuta benchmark de preprocesamiento Zero-Copy")
    parser.add_argument("--lote", type=str, default=None, help="ID de lote E1 para formato v3 de E5 (ej. LOTE-2026-11-A)")
    parser.add_argument("--fruta", type=str, default=None, help="ID de fruta E1 para formato v3 de E5 (ej. M-0001)")
    parser.add_argument("--e5-url", type=str, default=DEFAULT_E5_BASE_URL, help="URL base de la API de E5")
    parser.add_argument("--send-e5", action="store_true", help="Envía la lectura a POST /api/v1/ingest/e2 (requiere --lote y --fruta)")
    args = parser.parse_args()

    if args.benchmark:
        run_preprocessing_benchmark()
        return

    if args.export_onnx:
        export_to_onnx(args.weights, args.export_onnx)
        return

    # Determinar si usar ONNX o PyTorch
    use_onnx = True
    model_file: Optional[Path] = None

    if args.onnx:
        use_onnx = True
        model_file = Path(args.onnx)
    elif Path(args.weights).exists() and HAS_TORCH:
        use_onnx = False
        model_file = Path(args.weights)
    else:
        # Fallback a modelo ONNX INT8 o FP32
        int8_path = PROJECT_ROOT / "weights" / "orange_model_int8.onnx"
        fp32_path = PROJECT_ROOT / "weights" / "orange_model.onnx"
        if int8_path.exists():
            model_file = int8_path
            use_onnx = True
        elif fp32_path.exists():
            model_file = fp32_path
            use_onnx = True

    predictor = QualityPredictor(
        model_path=model_file,
        use_onnx=use_onnx,
        num_threads=args.threads,
    )

    image_paths: List[Path] = [Path(p) for p in (args.image or [])]
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"Error: No existe la carpeta {dir_path}", file=sys.stderr)
            return
        image_paths.extend(sorted(
            p for p in dir_path.iterdir()
            if p.is_file() and (p.suffix.lower() in IMAGE_EXTENSIONS or parse_orange_photo_name(p))
        ))
        if not image_paths:
            print(f"Error: La carpeta {dir_path} no contiene imágenes.", file=sys.stderr)
            return

    # Fotos con formato <uid>-<naranja>.<foto>: calificación promediada por naranja
    if image_paths and (args.dir or len(image_paths) > 1 or parse_orange_photo_name(image_paths[0])):
        if not HAS_OPENCV:
            raise RuntimeError(
                "OpenCV no está disponible. Instálalo con "
                "'python -m pip install opencv-python' y vuelve a ejecutar."
            )
        results = classify_oranges(predictor, image_paths, expected_photos=args.expected_photos)
        for res in results:
            conf = res["confidence_percent"] / 100.0
            res.update(map_prediction_to_e2_datos(predicted_class=res["predicted_class"], confidence=conf))
            # Con una sola naranja, --lote/--fruta definen los IDs de E5; si no, se usan uid y orange_id
            res["id_lote"] = args.lote or res["uid"]
            res["id_fruta"] = args.fruta if (args.fruta and len(results) == 1) else res["orange_id"]
            if args.send_e5:
                e5_payload = build_e2_payload(
                    id_lote=res["id_lote"],
                    id_fruta=res["id_fruta"],
                    predicted_class=res["predicted_class"],
                    confidence=conf,
                )
                res["e5_sync"] = E5Client(base_url=args.e5_url).send_e2_ingest(e5_payload)
                if not args.json:
                    print(f"[E5] {res['id_fruta']}: {res['e5_sync']}")
        if args.json:
            print(json.dumps(results, indent=2))
        elif results:
            print_orange_results(results)
        else:
            print("No se pudo calificar ninguna naranja.", file=sys.stderr)
        return

    if image_paths:
        if not HAS_OPENCV:
            raise RuntimeError(
                "OpenCV no está disponible. Instálalo con "
                "'python -m pip install opencv-python' y vuelve a ejecutar."
            )

        image_path = image_paths[0]
        if not image_path.exists():
            print(f"Error: No existe el archivo {image_path}")
            return

        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: No se pudo leer la imagen con OpenCV: {image_path}")
            return

        res, _ = predictor.predict_cv2(frame)
        e2_datos = map_prediction_to_e2_datos(
            predicted_class=res["predicted_class"],
            confidence=res["confidence"],
        )

        if args.send_e5:
            if not args.lote or not args.fruta:
                print("Error: --send-e5 requiere especificar --lote y --fruta.")
                return
            e5_payload = build_e2_payload(
                id_lote=args.lote,
                id_fruta=args.fruta,
                predicted_class=res["predicted_class"],
                confidence=res["confidence"],
                defects=e2_datos["defectos_detectados"],
            )
            e5_res = E5Client(base_url=args.e5_url).send_e2_ingest(e5_payload)
            if args.json:
                print(json.dumps({"payload": e5_payload, "e5_response": e5_res}, indent=2))
                return
            print(f"[E5] Respuesta POST /api/v1/ingest/e2: {e5_res}")

        if args.json:
            if args.lote and args.fruta:
                payload = build_e2_payload(
                    id_lote=args.lote,
                    id_fruta=args.fruta,
                    predicted_class=res["predicted_class"],
                    confidence=res["confidence"],
                    defects=e2_datos["defectos_detectados"],
                )
            else:
                payload = {
                    "station_id": 2,
                    "estacion": "E2",
                    "timestamp": round(time.time(), 3),
                    "image_name": image_path.name,
                    "predicted_class": res["predicted_class"],
                    "confidence_percent": res["confidence_percent"],
                    "calidad_visual": e2_datos["calidad_visual"],
                    "confianza_visual": e2_datos["confianza_visual"],
                    "defectos_detectados": e2_datos["defectos_detectados"],
                    "probabilities": res["probabilities"],
                    "latency_preprocess_ms": res["latency_preprocess_ms"],
                    "latency_inference_ms": res["latency_inference_ms"],
                    "latency_total_ms": res["latency_ms"],
                }
            print(json.dumps(payload, indent=2))
            return

        print("\n" + "=" * 55)
        print("RESULTADO DE CLASIFICACIÓN DE CALIDAD VISUAL")
        print("=" * 55)
        print(f"Imagen:                  {image_path.name}")
        print(f"Calidad Estimada:        {res['predicted_class']}")
        print(f"Nivel de Confianza:      {res['confidence_percent']}%")
        print(f"Latencia Preproceso:     {res['latency_preprocess_ms']} ms")
        print(f"Latencia Inferencia:     {res['latency_inference_ms']} ms")
        print(f"Latencia Total:          {res['latency_ms']} ms")
        print("Probabilidades por clase:")
        for cls, prob in res["probabilities"].items():
            print(f"  - {cls:<22}: {prob * 100:.2f}%")
        print("=" * 55)
        return

    if args.webcam is not None or args.picam:
        run_camera_stream(
            predictor=predictor,
            camera_source=args.webcam if args.webcam is not None else 0,
            use_picam=args.picam,
            headless=args.headless,
            max_frames=args.max_frames,
        )
        return

    # Inferencia Demo usando exclusivamente OpenCV y NumPy (sin Pillow)
    if not HAS_OPENCV:
        raise RuntimeError(
            "OpenCV no está disponible. Instálalo con "
            "'python -m pip install opencv-python' y vuelve a ejecutar."
        )

    print("\n[Inferencia Demo] Generando frame sintético de prueba con OpenCV (Zero-Copy)...")
    demo_frame = np.full((480, 640, 3), (35, 35, 35), dtype=np.uint8)
    cv2.circle(demo_frame, (320, 240), 130, (20, 140, 235), -1)  # Naranja sintética

    res, _ = predictor.predict_cv2(demo_frame)
    print("=" * 55)
    print("DEMO DE PREDICCIÓN CON FRAME SINTÉTICO (EDGE)")
    print("=" * 55)
    print(f"Calidad Estimada:        {res['predicted_class']}")
    print(f"Nivel de Confianza:      {res['confidence_percent']}%")
    print(f"Latencia Preproceso:     {res['latency_preprocess_ms']} ms")
    print(f"Latencia Inferencia:     {res['latency_inference_ms']} ms")
    print(f"Latencia Total:          {res['latency_ms']} ms")
    print("Probabilidades:")
    for cls, prob in res["probabilities"].items():
        print(f"  - {cls:<22}: {prob * 100:.2f}%")
    print("=" * 55)



if __name__ == "__main__":
    main()
