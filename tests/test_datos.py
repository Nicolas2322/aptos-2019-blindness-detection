import cv2
import numpy as np
import pandas as pd
from conftest import fondo_sintetico

from aptos.datos import asignar_pliegues, dhash, grupos_duplicados


def test_pliegues_estratificados_conservan_proporciones():
    y = np.repeat([0, 1, 2, 3, 4], [1805, 370, 999, 193, 295])  # distribución real de APTOS 2019
    df = asignar_pliegues(pd.DataFrame({"diagnosis": y}))
    assert sorted(df.pliegue.unique()) == [0, 1, 2, 3, 4]
    global_ = df.diagnosis.value_counts(normalize=True).sort_index()
    for _, g in df.groupby("pliegue"):
        assert np.abs(g.diagnosis.value_counts(normalize=True).sort_index() - global_).max() < 0.01


def test_duplicados_quedan_en_el_mismo_pliegue():
    df = pd.DataFrame({"diagnosis": [0, 0, 2, 2, 4, 4, 1, 3, 0, 2] * 10})
    grupos = pd.Series(np.arange(len(df)) // 2)  # cada par es un duplicado
    out = asignar_pliegues(df, grupos=grupos)
    assert (out.groupby(grupos.values).pliegue.nunique() == 1).all()


def test_dhash_detecta_duplicado_reescalado_y_distingue_otra_imagen():
    a = fondo_sintetico(semilla=3)
    a[200:260, 300:380] = 255  # lesión brillante
    b = cv2.resize(a, (400, 300))  # misma imagen a otra resolución
    c = fondo_sintetico(semilla=4); c[:, :400] = c[:, :400] // 3; c[80:300, 450:700] = 255  # otra retina, otra iluminación
    g = grupos_duplicados([dhash(a), dhash(b), dhash(c)])
    assert g[0] == g[1] and g[0] != g[2]
