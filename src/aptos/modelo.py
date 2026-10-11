"""Modelo: EfficientNet-B3 preentrenado en ImageNet con cabeza de regresión (sección 4 del diseño)."""
from __future__ import annotations


def crear_modelo(arquitectura: str = "efficientnet_b3", preentrenado: bool = True, abandono: float = 0.3):
    import timm
    return timm.create_model(arquitectura, pretrained=preentrenado, num_classes=1, drop_rate=abandono)


def congelar_extractor(modelo, congelar: bool = True) -> None:
    """Fase 1 del entrenamiento: solo se entrena la cabeza durante las primeras épocas."""
    for nombre, p in modelo.named_parameters():
        if not nombre.startswith("classifier"):
            p.requires_grad = not congelar
