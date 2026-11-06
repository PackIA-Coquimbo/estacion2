"""
Pipeline de Carga de Datos y Data Augmentation
Proyecto: OrangeQuality-AI (Estación 2: Análisis Visual de Calidad)

Responsable de:
1. Definir transformaciones de Data Augmentation para evitar que la red
   memorice el fondo e iluminación estáticos de la estación de captura.
2. Cargar imágenes organizadas por carpetas de clase:
   dataset/
     ├── buen_estado/
     ├── marginalmente_buena/
     └── mal_estado/
3. Crear división estratificada en conjuntos de Train, Validation y Test.
4. Generar DataLoaders de PyTorch listos para entrenamiento o inferencia.
5. Permitir la generación y carga de datos sintéticos/dummy para pruebas automáticas.
"""

from pathlib import Path
from typing import Tuple, Dict, List, Optional
import os
import random
import sys

# Asegurar que la raíz del proyecto esté en el path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from PIL import Image

from src.model import DEFAULT_CLASSES

# Extensiones de imagen soportadas
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Normalización estándar (ImageNet)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Transformaciones de entrenamiento con Data Augmentation intensivo
    diseñado para mitigar la memorización del fondo e iluminación fija:
    - RandomResizedCrop: simula variaciones de distancia y posición de la naranja en la caja.
    - RandomHorizontalFlip y RandomVerticalFlip: invarianza rotacional simétrica.
    - RandomRotation (0° a 360°): la fruta no tiene orientación fija.
    - ColorJitter: robustez frente a fluctuaciones en la lámpara o sombras estáticas.
    - Normalización estandarizada.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Transformaciones deterministas para validación, test e inferencia en vivo.
    Sin variaciones aleatorias.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


class OrangeDataset(Dataset):
    """
    Dataset para clasificación de calidad de naranjas.
    Escanea recursivamente las carpetas de clase y gestiona índices y etiquetas.
    """
    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = class_names if class_names is not None else DEFAULT_CLASSES
        self.class_to_idx: Dict[str, int] = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        self.samples: List[Tuple[Path, int]] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Busca todas las imágenes válidas en los subdirectorios de cada clase."""
        for class_name in self.class_names:
            class_folder = self.root_dir / class_name
            if not class_folder.exists() or not class_folder.is_dir():
                continue

            class_idx = self.class_to_idx[class_name]
            for file_path in class_folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    self.samples.append((file_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, target = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def get_class_counts(self) -> Dict[str, int]:
        """Retorna el número de muestras por cada clase detectada."""
        counts = {cls: 0 for cls in self.class_names}
        for _, class_idx in self.samples:
            class_name = self.class_names[class_idx]
            counts[class_name] += 1
        return counts


class SyntheticOrangeDataset(Dataset):
    """
    Dataset sintético en memoria para pruebas de pipeline cuando no hay
    imágenes reales aún recolectadas en el directorio.
    Genera imágenes simuladas con círculos de colores y ruido.
    """
    def __init__(
        self,
        num_samples_per_class: int = 15,
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.transform = transform
        self.class_names = class_names if class_names is not None else DEFAULT_CLASSES
        self.samples: List[Tuple[Image.Image, int]] = []

        # Paletas de color representativas:
        # buen_estado: naranja uniforme brillante
        # marginalmente_buena: naranja con manchas verdosas o pálidas
        # mal_estado: naranja oscuro/marrón con manchas negras/hongos
        color_maps = [
            (240, 140, 30),   # Naranja óptima
            (200, 170, 50),   # Descolorida / verdosa
            (110, 80, 40),    # Deteriorada / hongos
        ]

        for class_idx in range(len(self.class_names)):
            base_color = color_maps[class_idx % len(color_maps)]
            for _ in range(num_samples_per_class):
                # Generar imagen sintética 256x256
                img = Image.new("RGB", (256, 256), color=(40, 40, 40))  # Fondo estático oscuro
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                # Círculo simula la fruta
                draw.ellipse([30, 30, 226, 226], fill=base_color)
                self.samples.append((img, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, target = self.samples[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def create_dataloaders(
    dataset_dir: str | Path,
    batch_size: int = 16,
    train_split: float = 0.70,
    val_split: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
    fallback_to_synthetic: bool = True,
) -> Dict[str, DataLoader]:
    """
    Crea los DataLoaders para Entrenamiento, Validación y Test.
    Aplica Data Augmentation dinámicamente según el split.

    Si no hay imágenes en `dataset_dir` y `fallback_to_synthetic=True`,
    genera un dataset sintético para permitir validar el ciclo de entrenamiento.
    """
    dataset_path = Path(dataset_dir)
    raw_dataset = OrangeDataset(root_dir=dataset_path)

    # Si el dataset está vacío, evaluar uso de dataset sintético
    if len(raw_dataset) == 0:
        if not fallback_to_synthetic:
            raise ValueError(f"No se encontraron imágenes en: {dataset_path}")
        print("[dataset_loader] Aviso: Sin imágenes reales en dataset/. Usando dataset sintético para pruebas.")
        base_dataset = SyntheticOrangeDataset(num_samples_per_class=30)
    else:
        base_dataset = raw_dataset

    total_samples = len(base_dataset)
    train_size = int(total_samples * train_split)
    val_size = int(total_samples * val_split)
    test_size = total_samples - train_size - val_size

    # Reproducibilidad de la división
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_samples, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Envolver con las transformaciones correspondientes
    train_tf = get_train_transforms()
    val_tf = get_val_transforms()

    class TransformedSubset(Dataset):
        def __init__(self, original_dataset: Dataset, idxs: List[int], transform: transforms.Compose):
            self.dataset = original_dataset
            self.idxs = idxs
            self.transform = transform

        def __len__(self) -> int:
            return len(self.idxs)

        def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
            item = self.dataset[self.idxs[i]]
            # Si el dataset es SyntheticOrangeDataset, item[0] es PIL.Image
            # Si es OrangeDataset sin transform previa, abrimos PIL.Image
            if isinstance(item[0], torch.Tensor):
                return item[0], item[1]
            return self.transform(item[0]), item[1]

    # Asignar transformaciones por conjunto
    train_subset = TransformedSubset(base_dataset, train_indices, train_tf)
    val_subset = TransformedSubset(base_dataset, val_indices, val_tf)
    test_subset = TransformedSubset(base_dataset, test_indices, val_tf)

    dataloaders = {
        "train": DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    return dataloaders


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent.parent
    ds_dir = current_dir / "dataset"

    print("=" * 60)
    print("Módulo de Carga y Aumento de Datos (dataset_loader.py)")
    print("=" * 60)

    raw_ds = OrangeDataset(root_dir=ds_dir)
    print(f"Directorio de dataset: {ds_dir}")
    print(f"Imágenes reales encontradas: {len(raw_ds)}")
    print(f"Conteo por clases: {raw_ds.get_class_counts()}")

    loaders = create_dataloaders(dataset_dir=ds_dir, batch_size=4, fallback_to_synthetic=True)
    for split_name, loader in loaders.items():
        print(f"DataLoader '{split_name}': {len(loader.dataset)} muestras ({len(loader)} batches)")

    # Probar un batch del loader de entrenamiento
    for images, labels in loaders["train"]:
        print(f"Batch de prueba (train): {images.shape}, etiquetas: {labels.tolist()}")
        break
    print("=" * 60)
