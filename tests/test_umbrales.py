import numpy as np

from aptos.umbrales import OptimizadorUmbrales, a_grados, qwk


def test_qwk_perfecto_y_penalizacion_cuadratica():
    y = np.array([0, 1, 2, 3, 4] * 20)
    assert qwk(y, y) == 1.0
    cerca, lejos = np.clip(y + 1, 0, 4), np.where(y == 0, 4, 0)
    assert qwk(y, cerca) > qwk(y, lejos)


def test_conversion_a_grados_respeta_limites():
    assert list(a_grados([-1, 0.4, 0.6, 1.6, 2.6, 3.6, 9])) == [0, 0, 1, 2, 3, 4, 4]


def test_optimizador_mejora_o_iguala_umbrales_iniciales():
    rng = np.random.default_rng(0)
    y = rng.choice(5, 2000, p=[.49, .10, .27, .05, .09])
    sesgo = np.array([0.3, 0.1, -0.2, -0.4, -0.5])  # salida de regresión sesgada hacia la media
    valores = y + sesgo[y] + rng.normal(0, .45, len(y))
    base = qwk(y, a_grados(valores))
    opt = OptimizadorUmbrales().fit(valores, y)
    assert qwk(y, opt.predict(valores)) >= base
    assert np.all(np.diff(opt.umbrales_) > 0)
