import numpy as np

from aptos.monitoreo import UMBRAL_ALERTA, evaluar_deriva, psi, psi_categorico


def test_psi_sin_deriva_es_bajo_y_con_deriva_alto():
    rng = np.random.default_rng(0)
    ref = rng.normal(100, 15, 5000)
    assert psi(ref, rng.normal(100, 15, 2000)) < 0.05
    assert psi(ref, rng.normal(130, 15, 2000)) > UMBRAL_ALERTA


def test_psi_categorico_detecta_cambio_en_distribucion_de_grados():
    ref = np.repeat([0, 1, 2, 3, 4], [49, 10, 27, 5, 9])
    assert psi_categorico(ref, ref) < 1e-6
    assert psi_categorico(ref, np.repeat([0, 1, 2, 3, 4], [80, 5, 10, 2, 3])) > UMBRAL_ALERTA


def test_evaluar_deriva_marca_alertas_por_variable():
    rng = np.random.default_rng(1)
    ref = {"brillo_medio": rng.normal(90, 10, 3000), "fraccion_retina": rng.normal(.6, .05, 3000)}
    act = {"brillo_medio": rng.normal(120, 10, 500), "fraccion_retina": rng.normal(.6, .05, 500)}
    r = evaluar_deriva(ref, act)
    assert r["brillo_medio"]["alerta"] and not r["fraccion_retina"]["alerta"]
