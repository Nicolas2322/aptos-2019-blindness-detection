"""Particionamiento estratificado y detección de imágenes duplicadas (fuga entre pliegues)."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SEMILLA = 42


def asignar_pliegues(etiquetas: pd.DataFrame, n_pliegues: int = 5, columna: str = "diagnosis",
                     grupos: pd.Series | None = None) -> pd.DataFrame:
    """Añade la columna `pliegue`. Si se dan grupos (duplicados), todo el grupo cae en el mismo pliegue."""
    df = etiquetas.reset_index(drop=True).copy()
    df["pliegue"] = -1
    clave = grupos.reset_index(drop=True) if grupos is not None else pd.Series(np.arange(len(df)))
    repr_ = df.assign(_g=clave).groupby("_g")[columna].agg(lambda s: s.mode().iloc[0])
    skf = StratifiedKFold(n_pliegues, shuffle=True, random_state=SEMILLA)
    for k, (_, te) in enumerate(skf.split(repr_.index, repr_.values)):
        df.loc[clave.isin(repr_.index[te]).values, "pliegue"] = k
    return df


def dhash(img: np.ndarray, lado: int = 8) -> int:
    """Hash perceptual por diferencias: robusto a cambios de tamaño y compresión."""
    gris = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (lado + 1, lado), interpolation=cv2.INTER_AREA)
    bits = (gris[:, 1:] > gris[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def grupos_duplicados(hashes: list[int], distancia_max: int = 4) -> np.ndarray:
    """Agrupa imágenes cuyo hash difiere en a lo sumo `distancia_max` bits (unión-búsqueda)."""
    padre = list(range(len(hashes)))
    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]; i = padre[i]
        return i
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if bin(hashes[i] ^ hashes[j]).count("1") <= distancia_max:
                padre[raiz(i)] = raiz(j)
    return np.array([raiz(i) for i in range(len(hashes))])
