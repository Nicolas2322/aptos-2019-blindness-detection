import cv2
import numpy as np
import pandas as pd
from conftest import fondo_sintetico

from aptos.datos import (
    agrupacion_fiable,
    asignar_pliegues,
    dhash,
    grupos_desde_carpeta,
    grupos_duplicados,
    hashes_desde_carpeta,
)


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


def test_grupos_desde_carpeta_agrupa_los_duplicados_leidos_de_disco(tmp_path):
    """Duplicado al estilo de APTOS: la misma fotografía reenviada tras una recompresión."""
    a = fondo_sintetico(semilla=3)
    a[200:260, 300:380] = 255
    c = fondo_sintetico(semilla=4); c[:, :400] = c[:, :400] // 3; c[80:300, 450:700] = 255
    cv2.imwrite(str(tmp_path / "a.png"), a)
    recomprimida = cv2.imdecode(cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 85])[1], cv2.IMREAD_COLOR)
    cv2.imwrite(str(tmp_path / "a_copia.png"), recomprimida)
    cv2.imwrite(str(tmp_path / "c.png"), c)
    g = grupos_desde_carpeta(["a", "a_copia", "c"], tmp_path)
    assert g[0] == g[1] and g[0] != g[2]


def test_grupos_desde_carpeta_avisa_si_falta_una_imagen(tmp_path):
    cv2.imwrite(str(tmp_path / "a.png"), fondo_sintetico())
    try:
        grupos_desde_carpeta(["a", "inexistente"], tmp_path)
    except FileNotFoundError as e:
        assert "inexistente" in str(e)
    else:
        raise AssertionError("se esperaba FileNotFoundError")


def test_agrupacion_fiable_rechaza_un_grupo_que_abarca_el_conjunto():
    """Lo observado con hash de 64 bits: un grupo con el 66 % de las imágenes no son duplicados."""
    sanos = pd.Series(list(range(3000)) + [3000, 3000])          # duplicados reales: un par
    degenerados = pd.Series([0] * 2436 + list(range(1, 1227)))   # un grupo gigante
    assert agrupacion_fiable(sanos)
    assert not agrupacion_fiable(degenerados)


def test_hash_de_256_bits_distingue_retinas_que_el_de_64_confunde():
    """Las fotografías de fondo de ojo comparten estructura global: 64 bits no bastan."""
    def retina(semilla):
        img = fondo_sintetico(semilla=semilla)
        rng = np.random.default_rng(semilla)
        for _ in range(8):                                        # lesiones propias de cada imagen
            cv2.circle(img, (int(rng.integers(320, 480)), int(rng.integers(220, 380))),
                       int(rng.integers(5, 13)), (250, 240, 240), -1)
        return img
    a, b = retina(11), retina(12)
    assert bin(dhash(a, 8) ^ dhash(b, 8)).count("1") <= 4         # indistinguibles con 64 bits
    assert bin(dhash(a, 16) ^ dhash(b, 16)).count("1") > 4        # separables con 256 bits


def test_hashes_desde_carpeta_devuelve_uno_por_imagen_en_orden(tmp_path):
    for cid, semilla in (("a", 1), ("b", 2), ("c", 3)):
        cv2.imwrite(str(tmp_path / f"{cid}.png"), fondo_sintetico(semilla=semilla))
    hs = hashes_desde_carpeta(["a", "b", "c"], tmp_path)
    assert len(hs) == 3 and all(isinstance(h, int) for h in hs)
    assert hs == hashes_desde_carpeta(["a", "b", "c"], tmp_path)  # determinista
