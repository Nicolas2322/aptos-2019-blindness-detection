import cv2
import numpy as np
import pytest
from conftest import fondo_sintetico

from aptos.preprocesamiento import (
    TAMANO,
    ImagenInvalida,
    estadisticas_imagen,
    normalizacion_graham,
    preparar_imagen,
    recorte_circular,
    redimensionar_con_relleno,
)


def test_recorte_elimina_marco_negro(fondo):
    rec = recorte_circular(fondo)
    assert rec.shape[0] == pytest.approx(501, abs=2) and rec.shape[1] == pytest.approx(501, abs=2)


def test_redimensionado_conserva_relacion_de_aspecto():
    img = np.full((200, 400, 3), 100, np.uint8)
    out = redimensionar_con_relleno(img, 384)
    assert out.shape == (384, 384, 3)
    assert out[0, 192].sum() == 0 and out[192, 192].sum() > 0  # relleno arriba, contenido al centro


def test_salida_con_forma_y_tipo_esperados(fondo):
    tensor, procesada = preparar_imagen(fondo)
    assert tensor.shape == (3, TAMANO, TAMANO) and tensor.dtype == np.float32
    assert procesada.shape == (TAMANO, TAMANO, 3) and procesada.dtype == np.uint8
    assert np.isfinite(tensor).all()


def test_graham_reduce_diferencias_de_iluminacion():
    oscura = fondo_sintetico(brillo=(90, 45, 20), semilla=1)
    clara = fondo_sintetico(brillo=(220, 130, 70), semilla=1)
    _, a = preparar_imagen(oscura); _, b = preparar_imagen(clara)
    dif_original = abs(float(oscura[oscura > 7].mean()) - float(clara[clara > 7].mean()))
    dif_procesada = abs(float(a.mean()) - float(b.mean()))
    assert dif_procesada < 0.2 * dif_original


def test_entrenamiento_e_inferencia_son_identicos(fondo):
    """Contrato: la misma imagen produce exactamente el mismo tensor en ambos trayectos."""
    t1, _ = preparar_imagen(fondo.copy()); t2, _ = preparar_imagen(fondo.copy())
    np.testing.assert_array_equal(t1, t2)


@pytest.mark.parametrize("img", [np.zeros((600, 800, 3), np.uint8), np.zeros((64, 64, 3), np.uint8),
                                 np.zeros((600, 800), np.uint8)])
def test_rechaza_imagenes_invalidas(img):
    with pytest.raises(ImagenInvalida):
        preparar_imagen(img)


def test_estadisticas_para_monitoreo(fondo):
    e = estadisticas_imagen(fondo)
    assert set(e) == {"alto", "ancho", "brillo_medio", "contraste", "fraccion_retina"}
    assert 0.3 < e["fraccion_retina"] < 0.45


def _fraccion_oscura(procesada, umbral=60):
    """Fracción de píxeles muy oscuros dentro del disco útil de la imagen procesada."""
    gris = cv2.cvtColor(procesada, cv2.COLOR_RGB2GRAY)
    disco = np.zeros(gris.shape, np.uint8)
    cv2.circle(disco, (gris.shape[1] // 2, gris.shape[0] // 2), int(0.95 * gris.shape[0] / 2), 1, -1)
    return float(((gris < umbral) & (disco == 1)).sum() / (disco == 1).sum())


@pytest.mark.parametrize("w,h,radio,descripcion", [
    (1000, 1000, 480, "círculo completo: el recorte sale cuadrado"),
    (2416, 1736, 1150, "círculo recortado arriba y abajo: el recorte sale apaisado"),
])
def test_graham_no_introduce_banda_oscura_segun_el_encuadre(w, h, radio, descripcion):
    """Regresión: la máscara circular de radio fijo metía dentro el relleno de `redimensionar_con_relleno`
    y el realce lo convertía en una banda de alto contraste. Como el encuadre depende de la cámara y no
    de la patología, el modelo podía leer el grado en esa banda (correlación espuria, reto R4)."""
    procesada = normalizacion_graham(redimensionar_con_relleno(recorte_circular(fondo_sintetico(h=h, w=w, radio=radio)), 384))
    assert _fraccion_oscura(procesada) < 0.01, descripcion
