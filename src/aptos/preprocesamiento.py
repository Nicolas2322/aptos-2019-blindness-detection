"""Preprocesamiento compartido entre entrenamiento e inferencia (sección 5.3 del diseño).

Una sola implementación evita el sesgo entre entrenamiento y servicio (training-serving skew).
"""
from __future__ import annotations

import cv2
import numpy as np

TAMANO = 384
MEDIA_IMAGENET = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESV_IMAGENET = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImagenInvalida(ValueError):
    """La imagen no contiene una región retiniana identificable."""


def validar(img: np.ndarray, lado_minimo: int = 128) -> None:
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise ImagenInvalida("Se esperaba una imagen RGB de tres canales.")
    if min(img.shape[:2]) < lado_minimo:
        raise ImagenInvalida(f"La imagen debe medir al menos {lado_minimo} píxeles por lado.")


def recorte_circular(img: np.ndarray, tolerancia: int = 7, fraccion_minima: float = 0.05) -> np.ndarray:
    """Elimina el marco negro alrededor de la retina mediante umbralización de intensidad."""
    gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mascara = gris > tolerancia
    if mascara.mean() < fraccion_minima:
        raise ImagenInvalida("No se identificó una región retiniana en la imagen.")
    filas, cols = np.where(mascara.any(axis=1))[0], np.where(mascara.any(axis=0))[0]
    return img[filas[0]: filas[-1] + 1, cols[0]: cols[-1] + 1]


def redimensionar_con_relleno(img: np.ndarray, tamano: int = TAMANO) -> np.ndarray:
    """Lleva la imagen a tamano × tamano conservando la relación de aspecto."""
    h, w = img.shape[:2]
    escala = tamano / max(h, w)
    nueva = cv2.resize(img, (max(1, round(w * escala)), max(1, round(h * escala))), interpolation=cv2.INTER_AREA)
    lienzo = np.zeros((tamano, tamano, 3), dtype=img.dtype)
    y0, x0 = (tamano - nueva.shape[0]) // 2, (tamano - nueva.shape[1]) // 2
    lienzo[y0: y0 + nueva.shape[0], x0: x0 + nueva.shape[1]] = nueva
    return lienzo


def normalizacion_graham(img: np.ndarray, sigma_relativo: float = 30 / 512,
                         tolerancia: int = 7, margen: int = 6) -> np.ndarray:
    """Resta una versión suavizada de la imagen para realzar el contraste local (Graham, 2015).

    La máscara se deriva de la propia imagen y no de un círculo de radio fijo. Un círculo fijo
    incluye el relleno negro que añade `redimensionar_con_relleno` cuando el recorte no es cuadrado,
    y el realce lo convierte en una banda oscura de alto contraste: una característica espuria que
    depende del encuadre de la cámara y no de la patología.
    """
    sigma = sigma_relativo * img.shape[0]
    suavizada = cv2.GaussianBlur(img, (0, 0), sigma)
    salida = cv2.addWeighted(img, 4, suavizada, -4, 128)
    mascara = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > tolerancia).astype(np.uint8)
    # se erosiona para descartar el halo que el realce produce en el borde de la retina
    mascara = cv2.erode(mascara, np.ones((margen, margen), np.uint8), iterations=1)
    return (salida * mascara[..., None] + 128 * (1 - mascara[..., None])).astype(np.uint8)


def preparar_imagen(img: np.ndarray, tamano: int = TAMANO) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (tensor normalizado C×H×W float32, imagen procesada uint8 para visualizar)."""
    validar(img)
    procesada = normalizacion_graham(redimensionar_con_relleno(recorte_circular(img), tamano))
    tensor = (procesada.astype(np.float32) / 255.0 - MEDIA_IMAGENET) / DESV_IMAGENET
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)), procesada


def estadisticas_imagen(img: np.ndarray, tolerancia: int = 7) -> dict[str, float]:
    """Estadísticas que alimentan el monitoreo de deriva de datos (sección 6.7)."""
    gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    retina = gris > tolerancia
    return {
        "alto": float(img.shape[0]), "ancho": float(img.shape[1]),
        "brillo_medio": float(gris[retina].mean()) if retina.any() else 0.0,
        "contraste": float(gris[retina].std()) if retina.any() else 0.0,
        "fraccion_retina": float(retina.mean()),
    }
