"""
Módulo de Arquitectura de Red Neuronal Convolucional (CNN) desde Cero.
(Estación 2: Análisis Visual de Calidad)

Diseñado para clasificación de calidad visual de naranjas en 3 clases:
- buen_estado
- marginalmente_buena
- mal_estado
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Clases por defecto del proyecto según la estructura de dataset/
DEFAULT_CLASSES: List[str] = [
    "buen_estado",
    "marginalmente_buena",
    "mal_estado",
]


class ConvBlock(nn.Module):
    """
    Bloque convolucional doble con Batch Normalization y activación no lineal.
    """
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class OrangeQualityCNN(nn.Module):
    """
    Red Neuronal Convolucional personalizada creada desde cero para clasificar la
    calidad visual de naranjas a partir de imperfecciones, color y textura.

    Características:
    - 4 etapas de extracción de características jerárquicas (32 -> 64 -> 128 -> 256 filtros).
    - Batch Normalization para estabilizar el entrenamiento.
    - AdaptiveAvgPool2d para admitir resoluciones flexibles (típico 224x224).
    - Dropout para mitigar el sobreajuste al fondo o iluminación fija.
    - Cabeza clasificadora adaptable a `num_classes` (por defecto 3).
    """

    def __init__(
        self,
        num_classes: int = len(DEFAULT_CLASSES),
        in_channels: int = 3,
        dropout_rate: float = 0.3,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.class_names = class_names if class_names is not None else DEFAULT_CLASSES

        # Extractor de características convolucional (Backbone propio)
        self.features = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=32, pool=True),    # -> (B, 32, H/2, W/2)
            ConvBlock(in_channels=32, out_channels=64, pool=True),            # -> (B, 64, H/4, W/4)
            ConvBlock(in_channels=64, out_channels=128, pool=True),          # -> (B, 128, H/8, W/8)
            ConvBlock(in_channels=128, out_channels=256, pool=True),         # -> (B, 256, H/16, W/16)
        )

        # Reducción espacial invariante a resolución
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))                        # -> (B, 256, 1, 1)

        # Cabeza de clasificación completamente conectada
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )

        # Inicialización de pesos Kaiming / He para convergencia más rápida
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Inicializa los pesos convolucionales y lineales usando He normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor de entrada con forma (B, C, H, W)
        Returns:
            Logits con forma (B, num_classes)
        """
        x = self.features(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidades normalizadas vía Softmax (B, num_classes)."""
        self.eval()
        logits = self(x)
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Retorna:
            - Clases predichas (indices: Tensor[B])
            - Confianzas (probabilidad máxima: Tensor[B])
            - Nombres de etiquetas predichas (List[str] de tamaño B)
        """
        probs = self.predict_proba(x)
        confs, preds = torch.max(probs, dim=1)
        labels = [
            self.class_names[idx.item()] if idx.item() < len(self.class_names) else f"clase_{idx.item()}"
            for idx in preds
        ]
        return preds, confs, labels

    def count_parameters(self) -> Dict[str, int]:
        """Calcula el número de parámetros entrenables y totales."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def build_model(
    num_classes: int = len(DEFAULT_CLASSES),
    in_channels: int = 3,
    dropout_rate: float = 0.3,
    class_names: Optional[List[str]] = None,
) -> OrangeQualityCNN:
    """
    Función constructora del modelo creado desde cero.
    """
    return OrangeQualityCNN(
        num_classes=num_classes,
        in_channels=in_channels,
        dropout_rate=dropout_rate,
        class_names=class_names,
    )


if __name__ == "__main__":
    model = build_model()
    params = model.count_parameters()
    print("=" * 60)
    print("Arquitectura OrangeQualityCNN (Creada desde cero)")
    print("=" * 60)
    print(f"Clases objetivo ({model.num_classes}): {model.class_names}")
    print(f"Parámetros entrenables: {params['trainable']:,}")
    print(f"Parámetros totales:     {params['total']:,}")
    
    # Prueba con tensor dummy
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    preds, confs, labels = model.predict(dummy_input)
    print(f"Entrada dummy: {tuple(dummy_input.shape)}")
    print(f"Salida logits: {tuple(logits.shape)}")
    print(f"Predicciones de prueba: {labels} con confianzas {[round(c.item(), 4) for c in confs]}")
    print("=" * 60)
