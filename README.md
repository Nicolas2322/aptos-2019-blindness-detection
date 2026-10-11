# APTOS 2019 Blindness Detection — Aplicaciones en Deep Learning

[![CI](https://github.com/Nicolas2322/aptos-2019-blindness-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Nicolas2322/aptos-2019-blindness-detection/actions/workflows/ci.yml)

**Universidad de Cundinamarca · Especialización en Inteligencia Artificial**
Asignatura: Aplicaciones en Deep Learning · Docente: Wilson Mauro Rojas Reales
Estudiante: Yohan Nicolás Gómez Castañeda

Clasificación de la severidad de la retinopatía diabética (grados 0–4) a partir de fotografías de fondo de ojo,
con EfficientNet-B3 y cabeza de regresión, umbrales optimizados para el kappa cuadrático (QWK) y operación MLOps sobre Azure ML.

## Estructura

```
src/aptos/
  preprocesamiento.py   Recorte circular, redimensión con relleno y normalización de Graham (módulo compartido)
  umbrales.py           QWK y optimización de umbrales (Nelder-Mead)
  datos.py              Pliegues estratificados y detección de duplicados por hash perceptual
  monitoreo.py          Deriva de datos y de predicciones (PSI)
  modelo.py             EfficientNet-B3 con cabeza de regresión (timm)
  entrenar.py           Entrenamiento por validación cruzada con MLflow
app/app.py              Interfaz Gradio y registro estructurado de predicciones
azureml/                Pipeline de entrenamiento y despliegue administrado
scripts/puerta_calidad.py  Puerta de calidad antes del registro del modelo
.github/workflows/      CI (pruebas), CT (entrenamiento continuo) y CD (despliegue azul/verde)
```

## Uso

```bash
pip install -r requirements-ci.txt
pytest --cov=src/aptos            # pruebas unitarias y de contrato
pip install -r requirements.txt
python -m aptos.entrenar --config configs/entrenamiento.yaml --datos <carpeta de APTOS 2019>
```

## Entregas

- [x] R2-A1-S6 Informe de investigación
- [x] R2-A1-S7 Diseño de la solución
- [x] R2-A2-S8 Avance del proyecto I: cronograma, retos y diseño MLOps
- [ ] R2-A2-S9 Avance del proyecto II
- [ ] Entrega final
