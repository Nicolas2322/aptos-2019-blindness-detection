import numpy as np
import pytest


def fondo_sintetico(h=600, w=800, radio=250, brillo=(170, 90, 40), semilla=0):
    """Imagen tipo fondo de ojo: disco anaranjado con textura sobre marco negro."""
    rng = np.random.default_rng(semilla)
    img = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[:h, :w]
    disco = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= radio ** 2
    for c, b in enumerate(brillo):
        img[..., c][disco] = np.clip(b + rng.normal(0, 8, disco.sum()), 0, 255)
    return img


@pytest.fixture
def fondo():
    return fondo_sintetico()
