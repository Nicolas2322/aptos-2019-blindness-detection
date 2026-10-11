"""Puerta de calidad (P6): solo registra el modelo si supera el umbral y a la versión en producción."""
import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--resultados", required=True); ap.add_argument("--qwk-minimo", type=float, default=0.85)
ap.add_argument("--qwk-produccion", type=float, default=0.0); ap.add_argument("--modelo", required=True)
a = ap.parse_args()
r = json.load(open(a.resultados))
aprobado = r["qwk_media"] >= a.qwk_minimo and r["qwk_media"] > a.qwk_produccion and r["qwk_desv"] <= 0.03
print(json.dumps({"qwk_media": r["qwk_media"], "qwk_desv": r["qwk_desv"], "aprobado": aprobado}))
if not aprobado:
    sys.exit("Puerta de calidad no superada: el modelo no se registra.")
print(f"Modelo {a.modelo} aprobado para registro con umbrales {r['umbrales']}")
