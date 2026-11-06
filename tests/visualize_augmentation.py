"""
Script de Verificación e Inspección Visual de Data Augmentation (Hito 2.3)
Proyecto: OrangeQuality-AI (Estación 2: Análisis Visual de Calidad)

Toma muestras representativas de las 3 clases (buen_estado, marginalmente_buena, mal_estado)
y aplica el pipeline de aumento de src/dataset_loader.py.

Genera una matriz comparativa guardada en 'tests/augmentation_preview.png':
- Fila 1: buen_estado (Original + 3 variantes aumentadas)
- Fila 2: marginalmente_buena (Original + 3 variantes aumentadas)
- Fila 3: mal_estado (Original + 3 variantes aumentadas)
"""

from pathlib import Path
from typing import Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.dataset_loader import (
    get_train_transforms,
    NORM_MEAN,
    NORM_STD,
    VALID_IMAGE_EXTENSIONS,
)
from src.model import DEFAULT_CLASSES


def denormalize_tensor(tensor: torch.Tensor) -> Image.Image:
    """Convierte un tensor normalizado (3, H, W) de vuelta a PIL.Image RGB."""
    mean = np.array(NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(NORM_STD, dtype=np.float32).reshape(3, 1, 1)

    np_img = tensor.cpu().numpy()
    np_img = (np_img * std) + mean
    np_img = np.clip(np_img * 255.0, 0, 255).astype(np.uint8)
    np_img = np.transpose(np_img, (1, 2, 0))  # (H, W, 3)

    return Image.fromarray(np_img)


def find_sample_per_class(dataset_dir: Path) -> Dict[str, Path]:
    """Busca una imagen de ejemplo para cada clase."""
    samples = {}
    for cls in DEFAULT_CLASSES:
        cls_dir = dataset_dir / cls
        if cls_dir.exists() and cls_dir.is_dir():
            for f in sorted(cls_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    samples[cls] = f
                    break
    if not samples:
        raise FileNotFoundError("No se encontraron imágenes válidas en dataset/.")
    return samples


def generate_multi_class_preview(
    dataset_dir: Path,
    output_path: Path,
    num_variants_per_class: int = 3,
    image_size: int = 224,
) -> Path:
    print("=" * 70)
    print("INSPECCIÓN VISUAL DE DATA AUGMENTATION POR CLASE (HITO 2.3)")
    print("=" * 70)

    samples = find_sample_per_class(dataset_dir)
    pipeline = get_train_transforms(image_size=image_size)

    cols = 1 + num_variants_per_class  # 1 Original + N variantes
    rows = len(samples)
    margin = 16
    label_h = 28
    header_h = 50

    cell_w = image_size
    cell_h = image_size + label_h

    canvas_w = (cols * cell_w) + ((cols + 1) * margin)
    canvas_h = header_h + (rows * cell_h) + ((rows + 1) * margin)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(25, 27, 30))
    draw = ImageDraw.Draw(canvas)

    # Título principal
    draw.text(
        (margin, 15),
        "OrangeQuality-AI - Matriz de Data Augmentation por Categoría (224x224)",
        fill=(240, 240, 240),
    )

    for row_idx, (cls_name, img_path) in enumerate(samples.items()):
        print(f"[CLASE: {cls_name}] Usando muestra: {img_path.name}")
        raw_img = Image.open(img_path).convert("RGB")
        original_resized = raw_img.resize((image_size, image_size))

        variants = []
        for _ in range(num_variants_per_class):
            tensor = pipeline(raw_img)
            variants.append(denormalize_tensor(tensor))

        row_tiles = [(f"{cls_name} (Original)", original_resized)] + [
            (f"Aumento #{i+1}", v_img) for i, v_img in enumerate(variants)
        ]

        y = header_h + margin + row_idx * (cell_h + margin)

        for col_idx, (label, tile_img) in enumerate(row_tiles):
            x = margin + col_idx * (cell_w + margin)
            canvas.paste(tile_img, (x, y))
            draw.text((x, y + image_size + 6), label, fill=(190, 190, 190))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print("-" * 70)
    print(f"[EXITO] Previsualización exportada a: {output_path}")
    print(f"Dimensiones de la matriz: {canvas.size}")
    print("=" * 70)
    return output_path


def main():
    ds_dir = PROJECT_ROOT / "dataset"
    out_file = PROJECT_ROOT / "tests" / "augmentation_preview.png"
    generate_multi_class_preview(ds_dir, out_file)


if __name__ == "__main__":
    main()
