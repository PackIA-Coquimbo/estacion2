"""
Script de Entrenamiento Optimizado para CUDA (NVIDIA RTX 5060) con Fallback a CPU (Estación 2: Análisis Visual de Calidad)

Características:
1. Detección automática de hardware (CUDA con Tensor Cores o CPU).
2. Precisión Mixta (AMP - Automatic Mixed Precision) para máxima velocidad en RTX 5060.
3. Optimizador AdamW con Regularización L2 y Scheduler dinámico.
4. Early Stopping para evitar sobreajuste.
5. Checkpointing del mejor modelo y guardado del historial de métricas.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import argparse
import json
import sys
import time

# Asegurar raíz del proyecto en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.model import build_model, DEFAULT_CLASSES
from src.dataset_loader import create_dataloaders


class EarlyStopping:
    """Detiene el entrenamiento si la métrica de validación no mejora tras cierto número de épocas."""
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """Ejecuta una época de entrenamiento con forward, backward y precisión mixta."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_cuda = device.type == "cuda"

    for batch_idx, (images, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=use_cuda)
        targets = targets.to(device, non_blocking=use_cuda)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """Evalúa el modelo en el conjunto de validación o test."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    use_cuda = device.type == "cuda"

    for batch_idx, (images, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=use_cuda)
        targets = targets.to(device, non_blocking=use_cuda)

        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100.0
    return epoch_loss, epoch_acc


def train(
    dataset_dir: Path,
    weights_dir: Path,
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    patience: int = 7,
    seed: int = 42,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Pipeline completo de entrenamiento con checkpointing."""
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 1. Detección de Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("INICIO DE EJECUCION DE ENTRENAMIENTO - ORANGEQUALITY-AI")
    print("=" * 70)
    print(f"[DISPOSITIVO] Modo seleccionado: {device.type.upper()}")
    if device.type == "cuda":
        print(f"[DISPOSITIVO] Adaptador GPU: {torch.cuda.get_device_name(0)}")
        print(f"[DISPOSITIVO] Memoria total disponible: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        torch.backends.cudnn.benchmark = True
    else:
        print("[DISPOSITIVO] Ejecucion en CPU/Grafica integrada.")

    # 2. Carga de Datos
    print("\n[DATOS] Inicializando particiones de datos estratificadas (Train/Val/Test)...")
    dataloaders, class_weights = create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        seed=seed,
        fallback_to_synthetic=True,
        return_weights=True,
    )
    print(f"[DATOS] Conjunto de Entrenamiento: {len(dataloaders['train'].dataset)} muestras")
    print(f"[DATOS] Conjunto de Validacion:    {len(dataloaders['val'].dataset)} muestras")
    print(f"[DATOS] Conjunto de Evaluacion:    {len(dataloaders['test'].dataset)} muestras")
    print(f"[DATOS] Pesos de ponderacion de clases calculados:")
    for cls, w in zip(DEFAULT_CLASSES, class_weights.tolist()):
        print(f"        - {cls:<20}: {w:.4f}")

    # 3. Inicialización del Modelo desde cero
    model = build_model(num_classes=len(DEFAULT_CLASSES)).to(device)
    total_params = model.count_parameters()["trainable"]
    print(f"\n[MODELO] Arquitectura: OrangeQualityCNN | Parametros entrenables: {total_params:,}")

    # 4. Criterio con compensación de desbalance, Optimizador y Precisión Mixta
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    early_stopping = EarlyStopping(patience=patience)

    # 5. Registro de Historial
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
    }

    best_val_loss = float("inf")
    best_model_path = weights_dir / "best_orange_model.pth"
    latest_model_path = weights_dir / "latest_checkpoint.pth"

    start_time = time.time()

    print(f"\n[ENTRENAMIENTO] Limite maximo de epocas configurado: {epochs}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, scaler, device, max_batches=max_batches
        )
        val_loss, val_acc = evaluate(
            model, dataloaders["val"], criterion, device, max_batches=max_batches
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rates"].append(current_lr)

        epoch_duration = time.time() - epoch_start

        # Guardar si es el mejor modelo
        is_best = val_loss < best_val_loss
        marker = " [MEJOR]" if is_best else ""
        if is_best:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "num_classes": len(DEFAULT_CLASSES),
                    "class_names": DEFAULT_CLASSES,
                },
                best_model_path,
            )

        # Checkpoint de la última época
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "class_names": DEFAULT_CLASSES,
            },
            latest_model_path,
        )

        print(
            f"Epoca [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc:5.1f}% | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc:5.1f}% | "
            f"LR: {current_lr:.1e}{marker}"
        )

        if early_stopping(val_loss):
            print(f"\n[INFO] Criterio de parada temprana (Early Stopping) alcanzado en epoca {epoch}.")
            break

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"[INFO] Ciclo de entrenamiento finalizado. Tiempo total: {total_time:.1f}s")
    print(f"[CHECKPOINT] Mejor punto de control registrado: {best_model_path} (Val Loss: {best_val_loss:.4f})")

    # Evaluación en Test Set
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_acc = evaluate(model, dataloaders["test"], criterion, device, max_batches=max_batches)
        print(f"[EVALUACION] Metricas en conjunto Test -> Loss: {test_loss:.4f} | Exactitud: {test_acc:.2f}%")
        history["test_loss"] = test_loss
        history["test_acc"] = test_acc

    # Guardar historial en JSON
    history_file = weights_dir / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[HISTORIAL] Registro exportado a: {history_file}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Entrenador de Calidad de Naranjas (OrangeQuality-AI)")
    parser.add_argument("--epochs", type=int, default=15, help="Número máximo de épocas")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamaño de lote")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate inicial")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Regularización de pesos (L2)")
    parser.add_argument("--patience", type=int, default=7, help="Paciencia para early stopping")
    parser.add_argument("--dry-run", action="store_true", help="Ejecuta una sola época de prueba rápida para validar el pipeline")
    parser.add_argument("--dataset-dir", type=str, default=str(PROJECT_ROOT / "dataset"), help="Ruta al dataset")
    parser.add_argument("--weights-dir", type=str, default=str(PROJECT_ROOT / "weights"), help="Ruta de pesos")
    args = parser.parse_args()

    epochs = 1 if args.dry_run else args.epochs
    max_batches = 5 if args.dry_run else None
    if args.dry_run:
        print("[AVISO] Modo Dry Run activado: ejecutando 1 época con 5 lotes de validación técnica.")

    train(
        dataset_dir=Path(args.dataset_dir),
        weights_dir=Path(args.weights_dir),
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        max_batches=max_batches,
    )


if __name__ == "__main__":
    main()