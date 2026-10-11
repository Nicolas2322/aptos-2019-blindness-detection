"""Script de puntuación del despliegue administrado de Azure ML (init/run)."""
import base64
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

from aptos.preprocesamiento import ImagenInvalida, estadisticas_imagen, preparar_imagen
from aptos.umbrales import a_grados

SESION, UMBRALES = None, None


def init():
    global SESION, UMBRALES
    carpeta = os.getenv("AZUREML_MODEL_DIR", ".")
    SESION = ort.InferenceSession(os.path.join(carpeta, "modelo.onnx"), providers=["CPUExecutionProvider"])
    UMBRALES = json.load(open(os.path.join(carpeta, "umbrales.json")))


def run(solicitud: str):
    datos = base64.b64decode(json.loads(solicitud)["input_data"])
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    try:
        tensor, _ = preparar_imagen(img)
    except ImagenInvalida as e:
        return {"error": str(e)}
    valor = float(SESION.run(None, {SESION.get_inputs()[0].name: tensor[None]})[0].squeeze())
    return {"grado": int(a_grados([valor], UMBRALES)[0]), "valor": valor, "estadisticas": estadisticas_imagen(img)}
