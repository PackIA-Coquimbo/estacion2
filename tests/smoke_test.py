"""
Smoke Test - Fase 1: Verificación de Arquitectura y Flujo de Tensores
(Estación 2: Calidad Visual de Naranjas)

Verifica que el modelo creado desde cero:
1. Se instancie correctamente con las 3 clases objetivo.
2. Admita tensores dummy sin discrepancia de dimensiones en forward y backward pass.
3. Genere probabilidades y etiquetas válidas.
4. Se ejecute de forma transparente en CPU o GPU integrada/dedicada.
5. Valide la existencia de los directorios clave del proyecto.
"""

import sys
from pathlib import Path

# Agregar raíz del proyecto al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from src.model import OrangeQualityCNN, build_model, DEFAULT_CLASSES


def test_directory_structure():
    """Verifica que la estructura de carpetas requerida exista."""
    print("[1/5] Verificando estructura de directorios y clases...")
    required_dirs = [
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "dataset" / "buen_estado",
        PROJECT_ROOT / "dataset" / "marginalmente_buena",
        PROJECT_ROOT / "dataset" / "mal_estado",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "weights",
        PROJECT_ROOT / "tests",
    ]
    for d in required_dirs:
        assert d.exists() and d.is_dir(), f"Directorio faltante: {d}"
    print("  -> Todos los directorios requeridos existen correctamente.")


def test_model_instantiation():
    """Verifica la instanciación de la CNN personalizada desde cero."""
    print("[2/5] Verificando instanciación del modelo desde cero...")
    model = build_model()
    assert isinstance(model, nn.Module), "El modelo debe ser una subclase de nn.Module."
    assert model.num_classes == 3, f"Se esperaban 3 clases, pero se obtuvieron {model.num_classes}."
    assert model.class_names == DEFAULT_CLASSES, f"Nombres de clase discordantes: {model.class_names}"

    params = model.count_parameters()
    assert params["trainable"] > 0, "El modelo debe tener parámetros entrenables."
    print(f"  -> Modelo instanciado con éxito.")
    print(f"  -> Clases: {model.class_names}")
    print(f"  -> Parámetros entrenables: {params['trainable']:,}")


def test_forward_pass():
    """Verifica el flujo del tensor (B, 3, 224, 224) y salida (B, num_classes)."""
    print("[3/5] Verificando forward pass con tensores dummy (CPU/Integrada)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.eval()

    # Probar con tamaño de batch 1 y 4
    for batch_size in (1, 4):
        dummy_tensor = torch.randn(batch_size, 3, 224, 224, device=device)
        with torch.no_grad():
            output = model(dummy_tensor)

        assert output.shape == (batch_size, 3), (
            f"Forma de salida incorrecta: {output.shape}, se esperaba ({batch_size}, 3)"
        )
        assert not torch.isnan(output).any(), "La salida contiene valores NaN."
        assert not torch.isinf(output).any(), "La salida contiene valores Infinitos."

    print(f"  -> Forward pass verificado en {device} para B=1 y B=4 sin errores.")


def test_predictions_and_probabilities():
    """Verifica que Softmax sume 1 y las etiquetas coincidan con las clases."""
    print("[4/5] Verificando cálculo de probabilidades y etiquetas...")
    model = build_model()
    dummy_tensor = torch.randn(3, 3, 224, 224)

    probs = model.predict_proba(dummy_tensor)
    assert probs.shape == (3, 3), f"Forma de probabilidades incorrecta: {probs.shape}"
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Las probabilidades deben sumar 1.0"

    preds, confs, labels = model.predict(dummy_tensor)
    assert len(labels) == 3, f"Se esperaban 3 etiquetas, obtenidas {len(labels)}"
    for label in labels:
        assert label in DEFAULT_CLASSES, f"Etiqueta desconocida retornada: {label}"

    print(f"  -> Probabilidades y etiquetas generadas correctamente: {labels}")


def test_backward_pass():
    """Verifica que el grafo de cómputo soporte retropropagación (backprop) sin fallos."""
    print("[5/5] Verificando backward pass (gradientes)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.train()

    dummy_tensor = torch.randn(2, 3, 224, 224, device=device)
    dummy_target = torch.tensor([0, 2], dtype=torch.long, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    outputs = model(dummy_tensor)
    loss = criterion(outputs, dummy_target)
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss), "La pérdida dio NaN."
    assert loss.item() > 0, "La pérdida calculada debe ser positiva."
    print(f"  -> Backward pass exitoso. Loss inicial: {loss.item():.4f}")


def run_all_tests():
    print("=" * 65)
    print("EJECUTANDO SMOKE TESTS - FASE 1 (OrangeQuality-AI)")
    print("=" * 65)
    test_directory_structure()
    test_model_instantiation()
    test_forward_pass()
    test_predictions_and_probabilities()
    test_backward_pass()
    print("=" * 65)
    print("TODOS LOS SMOKE TESTS PASARON EXITOSAMENTE (FASE 1 COMPLETADA)")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
