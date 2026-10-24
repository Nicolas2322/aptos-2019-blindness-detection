"""Diagnóstico del artefacto de encuadre (reto RE8).

Responde a una sola pregunta: la banda oscura que el preprocesamiento anterior introducía,
¿llevaba información sobre el grado de retinopatía?

Si la lleva, el descenso de QWK al corregir el preprocesamiento mide cuánto dependía el modelo
de un atajo ajeno a la patología. Si no la lleva, el descenso hay que buscarlo en otra parte y
la primera sospechosa es la erosión de la máscara nueva, que podría estar recortando retina.

Se ejecuta sin GPU. Uso:  python scripts/diagnostico_encuadre.py <carpeta_datos> [por_grado]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aptos.preprocesamiento import (  # noqa: E402
    TAMANO,
    ImagenInvalida,
    recorte_circular,
    redimensionar_con_relleno,
)

NEGRO = 7          # por debajo de este gris se considera relleno, no retina
MARGEN = 6         # el mismo que usa la máscara corregida


def graham_antiguo(img: np.ndarray, sigma_relativo: float = 30 / 512) -> np.ndarray:
    """Versión anterior al arreglo: máscara circular de radio fijo, que encerraba el relleno."""
    sigma = sigma_relativo * img.shape[0]
    suavizada = cv2.GaussianBlur(img, (0, 0), sigma)
    salida = cv2.addWeighted(img, 4, suavizada, -4, 128)
    mascara = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mascara, (img.shape[1] // 2, img.shape[0] // 2), int(0.95 * img.shape[0] / 2), 1, -1)
    return (salida * mascara[..., None] + 128 * (1 - mascara[..., None])).astype(np.uint8)


def disco(lado: int) -> np.ndarray:
    m = np.zeros((lado, lado), np.uint8)
    cv2.circle(m, (lado // 2, lado // 2), int(0.95 * lado / 2), 1, -1)
    return m


def medidas(ruta: Path) -> dict | None:
    """Mide el encuadre y la banda espuria de una imagen."""
    img = cv2.imread(str(ruta), cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        recorte = recorte_circular(img)
    except ImagenInvalida:
        return None
    alto, ancho = recorte.shape[:2]
    redimensionada = redimensionar_con_relleno(recorte, TAMANO)

    gris = cv2.cvtColor(redimensionada, cv2.COLOR_RGB2GRAY)
    relleno = (gris <= NEGRO)                      # lo que añadió el redimensionamiento
    d = disco(TAMANO).astype(bool)

    # Banda espuria: relleno que la máscara circular de radio fijo dejaba DENTRO de la zona realzada.
    banda = float((relleno & d).sum() / d.sum())

    # Retina que la erosión de la máscara nueva descarta, para detectar un recorte excesivo.
    retina = (~relleno).astype(np.uint8)
    erosionada = cv2.erode(retina, np.ones((MARGEN, MARGEN), np.uint8), iterations=1)
    perdida = float((retina.sum() - erosionada.sum()) / max(retina.sum(), 1))

    return {
        "aspecto_recorte": ancho / alto,
        "banda_espuria": banda,
        "retina_descartada_por_erosion": perdida,
        "brillo_retina": float(gris[~relleno].mean()) if (~relleno).any() else float("nan"),
    }


def auc(puntajes: np.ndarray, positivo: np.ndarray) -> float:
    """Área bajo la curva ROC por el estadístico de Mann-Whitney, sin dependencias extra."""
    rangos = pd.Series(puntajes).rank(method="average").to_numpy()   # promedia los empates
    n1, n0 = int(positivo.sum()), int((~positivo).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rangos[positivo].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    datos = Path(sys.argv[1])
    por_grado = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    etiquetas = pd.read_csv(datos / "train.csv")

    # --- 1. Relación de aspecto del archivo original: gratis, sobre el conjunto completo ---
    formas = []
    for cid in etiquetas.id_code:
        with Image.open(datos / "train_images" / f"{cid}.png") as im:
            formas.append(im.size)                      # (ancho, alto)
    etiquetas["aspecto_archivo"] = [w / h for w, h in formas]

    print("Relación de aspecto del ARCHIVO por grado (conjunto completo, n = "
          f"{len(etiquetas)}):")
    print(etiquetas.groupby("diagnosis").aspecto_archivo.agg(["mean", "std"]).round(4).to_string())
    rho_archivo = etiquetas.aspecto_archivo.corr(etiquetas.diagnosis, method="spearman")
    print(f"Correlación de Spearman con el grado: {rho_archivo:+.4f}\n")

    # --- 2. Banda espuria y encuadre sobre una muestra estratificada ---
    muestra = (etiquetas.sample(frac=1, random_state=42)
               .groupby("diagnosis").head(por_grado).reset_index(drop=True))
    filas = []
    for cid, g in zip(muestra.id_code, muestra.diagnosis):
        m = medidas(datos / "train_images" / f"{cid}.png")
        if m is not None:
            filas.append({**m, "diagnosis": int(g)})
    med = pd.DataFrame(filas)
    print(f"Muestra medida: {len(med)} imágenes "
          f"({med.diagnosis.value_counts().sort_index().tolist()} por grado)\n")

    resumen = med.groupby("diagnosis").agg(
        banda_media=("banda_espuria", "mean"),
        banda_desv=("banda_espuria", "std"),
        con_banda=("banda_espuria", lambda s: float((s > 0.01).mean())),
        aspecto=("aspecto_recorte", "mean"),
        erosion=("retina_descartada_por_erosion", "mean"),
        brillo=("brillo_retina", "mean"),
    ).round(4)
    print("Banda espuria del preprocesamiento ANTERIOR, por grado:")
    print(resumen.to_string())

    rho_banda = med.banda_espuria.corr(med.diagnosis, method="spearman")
    rho_asp = med.aspecto_recorte.corr(med.diagnosis, method="spearman")
    referible = (med.diagnosis >= 2).to_numpy()
    auc_banda = auc(med.banda_espuria.to_numpy(), referible)
    auc_aspecto = auc(med.aspecto_recorte.to_numpy(), referible)

    print(f"\nSpearman banda-grado:    {rho_banda:+.4f}")
    print(f"Spearman aspecto-grado:  {rho_asp:+.4f}")
    print(f"AUC de la banda sola para RD referible (grado >= 2):    {auc_banda:.4f}")
    print(f"AUC del aspecto solo para RD referible (grado >= 2):    {auc_aspecto:.4f}")
    print("\nLectura: un AUC cercano a 0,50 significa que la banda no distingue enfermos de sanos "
          "y por tanto no era un atajo aprovechable. Cuanto más se aleje de 0,50 —por arriba o por "
          "abajo— más información sobre el grado llevaba un artefacto que depende de la cámara.")
    print(f"\nRetina descartada por la erosión de la máscara nueva: "
          f"{med.retina_descartada_por_erosion.mean()*100:.2f} % de media "
          f"(máximo {med.retina_descartada_por_erosion.max()*100:.2f} %).")

    salida = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
    med.to_csv(salida / "diagnostico_encuadre.csv", index=False)
    json.dump({
        "n_muestra": int(len(med)),
        "rho_archivo_grado": float(rho_archivo),
        "rho_banda_grado": float(rho_banda),
        "rho_aspecto_grado": float(rho_asp),
        "auc_banda_referible": float(auc_banda),
        "auc_aspecto_referible": float(auc_aspecto),
        "erosion_media": float(med.retina_descartada_por_erosion.mean()),
        "banda_por_grado": resumen.banda_media.to_dict(),
        "con_banda_por_grado": resumen.con_banda.to_dict(),
    }, open(salida / "diagnostico_encuadre.json", "w"), indent=1, ensure_ascii=False)
    print(f"\nEscritos {salida}/diagnostico_encuadre.csv y .json")


if __name__ == "__main__":
    main()
