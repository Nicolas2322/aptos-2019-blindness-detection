import cv2
import numpy as np
import pandas as pd
from conftest import fondo_sintetico

from aptos.entrenar import construir_cache


def test_construir_cache_preprocesa_y_conserva_el_orden(tmp_path):
    origen, cache = tmp_path / "img", tmp_path / "cache"
    origen.mkdir()
    for cid in ("a", "b"):
        cv2.imwrite(str(origen / f"{cid}.png"), fondo_sintetico())
    df = pd.DataFrame({"id_code": ["a", "b"], "diagnosis": [0, 2]})
    rutas = construir_cache(df, origen, cache, 384)
    assert [r.stem for r in rutas] == ["a", "b"]
    assert cv2.imread(str(rutas[0])).shape == (384, 384, 3)


def test_construir_cache_no_aborta_ante_una_imagen_sin_retina(tmp_path, capsys):
    """APTOS contiene fotografías muy oscuras: una sola no debe detener el entrenamiento."""
    origen, cache = tmp_path / "img", tmp_path / "cache"
    origen.mkdir()
    cv2.imwrite(str(origen / "buena.png"), fondo_sintetico())
    cv2.imwrite(str(origen / "negra.png"), np.zeros((300, 400, 3), np.uint8))
    df = pd.DataFrame({"id_code": ["buena", "negra"], "diagnosis": [0, 4]})
    rutas = construir_cache(df, origen, cache, 384)
    assert len(rutas) == 2 and all(r.exists() for r in rutas)
    assert "negra" in capsys.readouterr().out
