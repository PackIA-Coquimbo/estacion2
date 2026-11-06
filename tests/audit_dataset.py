"""
Auditoría y Validación de Integridad del Dataset
Proyecto: OrangeQuality-AI (Estación 2: Análisis Visual de Calidad)

Verifica:
1. Legibilidad de cada imagen con PIL.
2. Formato, modo de color (RGB) y dimensiones.
3. Detección de archivos corruptos, truncados o vacíos (0 bytes).
4. Conteo por clase y cálculo de pesos recomendados para balanceo.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import sys
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_CLASSES = ["buen_estado", "marginalmente_buena", "mal_estado"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def audit_dataset(dataset_dir: Path) -> Dict[str, any]:
    print("=" * 70)
    print("AUDITORÍA DE INTEGRIDAD DEL DATASET - ORANGEQUALITY-AI")
    print(f"Ruta analizada: {dataset_dir}")
    print("=" * 70)

    stats = {
        "total_files": 0,
        "valid_images": 0,
        "corrupted_images": [],
        "invalid_extensions": [],
        "counts_per_class": {},
        "resolutions": {},
    }

    for class_name in DEFAULT_CLASSES:
        class_folder = dataset_dir / class_name
        if not class_folder.exists():
            print(f"[ALERTA] La carpeta de clase '{class_name}' no existe.")
            stats["counts_per_class"][class_name] = 0
            continue

        valid_in_class = 0
        for file_path in class_folder.iterdir():
            if not file_path.is_file():
                continue

            stats["total_files"] += 1

            if file_path.suffix.lower() not in VALID_EXTENSIONS:
                stats["invalid_extensions"].append(file_path)
                continue

            # Intentar abrir y verificar integridad de la imagen
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verifica que la estructura del archivo no esté rota

                # Reabrir para chequear modo y tamaño (verify() cierra o invalida el objeto en algunas versiones)
                with Image.open(file_path) as img:
                    res = f"{img.width}x{img.height}"
                    stats["resolutions"][res] = stats["resolutions"].get(res, 0) + 1
                    # Test de conversión a RGB
                    _ = img.convert("RGB")

                valid_in_class += 1
                stats["valid_images"] += 1

            except Exception as e:
                stats["corrupted_images"].append((file_path, str(e)))

        stats["counts_per_class"][class_name] = valid_in_class
        print(f"  • {class_name:<20}: {valid_in_class:>5} imágenes válidas")

    print("-" * 70)
    print(f"Total archivos escaneados: {stats['total_files']}")
    print(f"Total imágenes íntegras:   {stats['valid_images']}")

    if stats["corrupted_images"]:
        print(f"[PELIGRO] Se encontraron {len(stats['corrupted_images'])} imágenes corruptas:")
        for path, err in stats["corrupted_images"]:
            print(f"   - {path.name}: {err}")
    else:
        print("[OK] Ninguna imagen corrupta encontrada. 100% de archivos íntegros.")

    if stats["invalid_extensions"]:
        print(f"[AVISO] {len(stats['invalid_extensions'])} archivos con extensiones no soportadas.")
    else:
        print("[OK] No se encontraron archivos extraños o extensiones inválidas.")

    # Top resoluciones
    print("\nResoluciones detectadas (Top 5):")
    sorted_res = sorted(stats["resolutions"].items(), key=lambda x: x[1], reverse=True)[:5]
    for res, count in sorted_res:
        print(f"  - {res}: {count} imágenes ({count / max(stats['valid_images'], 1) * 100:.1f}%)")

    # Cálculo de pesos para balanceo (Inverse Class Frequency)
    total_valid = stats["valid_images"]
    num_classes = len(DEFAULT_CLASSES)
    weights = {}
    print("\nPesos sugeridos para ponderar CrossEntropyLoss (mitigación de desbalance):")
    for cls in DEFAULT_CLASSES:
        count = stats["counts_per_class"].get(cls, 0)
        if count > 0:
            # Fórmula estándar: N_total / (N_classes * N_class)
            w = total_valid / (num_classes * count)
            weights[cls] = round(w, 4)
            print(f"  - {cls:<20}: {w:.4f}  (Muestras: {count})")
        else:
            weights[cls] = 1.0

    print("=" * 70)
    return stats


if __name__ == "__main__":
    audit_dataset(DATASET_DIR)

