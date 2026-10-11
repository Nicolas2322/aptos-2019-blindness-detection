"""Entrenamiento por validación cruzada estratificada con registro en MLflow.

Uso (Kaggle o Azure ML con GPU):
    python -m aptos.entrenar --config configs/entrenamiento.yaml --datos /kaggle/input/aptos2019-blindness-detection
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from aptos.datos import asignar_pliegues
from aptos.preprocesamiento import (
    DESV_IMAGENET,
    MEDIA_IMAGENET,
    normalizacion_graham,
    recorte_circular,
    redimensionar_con_relleno,
)
from aptos.umbrales import OptimizadorUmbrales, a_grados, qwk


def fijar_semilla(semilla: int) -> None:
    import torch
    random.seed(semilla); np.random.seed(semilla); torch.manual_seed(semilla); torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.benchmark = False


def aumentos(cfg):
    import albumentations as A
    return A.Compose([
        A.RandomResizedCrop(size=(cfg["tamano"], cfg["tamano"]), scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), translate_percent=(-0.1, 0.1), p=0.7),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.7),
        A.HueSaturationValue(5, 10, 0, p=0.3),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 24), hole_width_range=(8, 24), p=0.3),
    ])


class ConjuntoFondo:
    """Lee imágenes ya preprocesadas en caché (recorte + redimensión + Graham) y aplica aumentos."""

    def __init__(self, rutas, etiquetas, transformacion=None):
        self.rutas, self.etiquetas, self.t = list(rutas), np.asarray(etiquetas, np.float32), transformacion

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        import torch
        img = cv2.cvtColor(cv2.imread(str(self.rutas[i])), cv2.COLOR_BGR2RGB)
        if self.t is not None:
            img = self.t(image=img)["image"]
        x = (img.astype(np.float32) / 255.0 - MEDIA_IMAGENET) / DESV_IMAGENET
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.tensor(self.etiquetas[i])


def construir_cache(df, carpeta_img: Path, carpeta_cache: Path, tamano: int) -> list[Path]:
    """Preprocesa una sola vez cada imagen: reduce el costo de E/S y de GPU por época (reto R3)."""
    carpeta_cache.mkdir(parents=True, exist_ok=True); rutas = []
    for cid in df.id_code:
        destino = carpeta_cache / f"{cid}.png"
        if not destino.exists():
            img = cv2.cvtColor(cv2.imread(str(carpeta_img / f"{cid}.png")), cv2.COLOR_BGR2RGB)
            proc = normalizacion_graham(redimensionar_con_relleno(recorte_circular(img), tamano))
            cv2.imwrite(str(destino), cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
        rutas.append(destino)
    return rutas


def entrenar_pliegue(k, df, rutas, cfg, dispositivo):
    import torch
    from torch.utils.data import DataLoader

    from aptos.modelo import congelar_extractor, crear_modelo
    tr, va = df.index[df.pliegue != k], df.index[df.pliegue == k]
    dl_tr = DataLoader(ConjuntoFondo([rutas[i] for i in tr], df.diagnosis[tr], aumentos(cfg)), cfg["lote"],
                       shuffle=True, num_workers=2, drop_last=True)
    dl_va = DataLoader(ConjuntoFondo([rutas[i] for i in va], df.diagnosis[va]), cfg["lote"] * 2, num_workers=2)
    modelo = crear_modelo(cfg["arquitectura"]).to(dispositivo)
    cabeza = [p for n, p in modelo.named_parameters() if n.startswith("classifier")]
    cuerpo = [p for n, p in modelo.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": cuerpo, "lr": cfg["lr_extractor"]}, {"params": cabeza, "lr": cfg["lr_cabeza"]}],
                            weight_decay=cfg["decaimiento"])
    total = cfg["epocas_congeladas"] + cfg["epocas_ajuste"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg["lr_extractor"], cfg["lr_cabeza"]],
                                                total_steps=total * len(dl_tr), pct_start=1 / total)
    escalador = torch.amp.GradScaler(enabled=dispositivo == "cuda"); perdida = torch.nn.MSELoss()
    mejor, sin_mejora, pred_mejor = -1.0, 0, None
    for ep in range(total):
        congelar_extractor(modelo, ep < cfg["epocas_congeladas"]); modelo.train()
        for j, (x, y) in enumerate(dl_tr):
            with torch.autocast(dispositivo, enabled=dispositivo == "cuda"):
                loss = perdida(modelo(x.to(dispositivo)).squeeze(1), y.to(dispositivo)) / cfg["acumulacion"]
            escalador.scale(loss).backward()
            if (j + 1) % cfg["acumulacion"] == 0:
                escalador.step(opt); escalador.update(); opt.zero_grad()
            sched.step()
        modelo.eval(); pred = []
        with torch.no_grad():
            for x, _ in dl_va:
                x = x.to(dispositivo)
                pred.append(((modelo(x) + modelo(torch.flip(x, dims=[3]))) / 2).squeeze(1).float().cpu().numpy())  # TTA
        pred = np.concatenate(pred); q = qwk(df.diagnosis[va], a_grados(pred))
        import mlflow
        mlflow.log_metric(f"qwk_val_pliegue{k}", q, step=ep)
        if q > mejor:
            mejor, sin_mejora, pred_mejor = q, 0, pred
            torch.save(modelo.state_dict(), f"modelo_pliegue{k}.pt")
        else:
            sin_mejora += 1
            if sin_mejora >= cfg["paciencia"]:
                break
    return va, pred_mejor, mejor


def main():
    import mlflow
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--datos", required=True)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); fijar_semilla(cfg["semilla"])
    datos = Path(a.datos); df = asignar_pliegues(pd.read_csv(datos / "train.csv"), cfg["pliegues"])
    rutas = construir_cache(df, datos / "train_images", Path("cache"), cfg["tamano"])
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    mlflow.set_experiment("aptos2019")
    with mlflow.start_run(run_name=cfg["arquitectura"]):
        mlflow.log_params(cfg)
        oof = np.zeros(len(df)); qwks = []
        for k in range(cfg["pliegues"]):
            va, pred, q = entrenar_pliegue(k, df, rutas, cfg, dispositivo); oof[va] = pred; qwks.append(q)
        opt = OptimizadorUmbrales().fit(oof, df.diagnosis)
        res = {"qwk_media": float(np.mean(qwks)), "qwk_desv": float(np.std(qwks)),
               "qwk_oof_umbrales": qwk(df.diagnosis, opt.predict(oof)), "umbrales": opt.umbrales_.tolist()}
        mlflow.log_metrics({k: v for k, v in res.items() if k != "umbrales"}); mlflow.log_dict(res, "resultados.json")
        json.dump(res, open("resultados.json", "w"), indent=1); print(res)


if __name__ == "__main__":
    main()
