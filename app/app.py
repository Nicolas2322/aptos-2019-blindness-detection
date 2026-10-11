"""Interfaz Gradio y punto de acceso programático (Anexo A del informe de diseño)."""
import json
import os
import time

import gradio as gr
import numpy as np

from aptos.preprocesamiento import ImagenInvalida, estadisticas_imagen, preparar_imagen
from aptos.umbrales import a_grados

GRADOS = {0: "0 - Sin retinopatía", 1: "1 - RD no proliferativa leve", 2: "2 - RD no proliferativa moderada",
          3: "3 - RD no proliferativa severa", 4: "4 - RD proliferativa"}
UMBRALES = json.loads(os.getenv("UMBRALES", "[0.5, 1.5, 2.5, 3.5]"))
VERSION = os.getenv("VERSION_MODELO", "dev")


def cargar_modelo():
    import onnxruntime as ort
    return ort.InferenceSession(os.getenv("RUTA_MODELO", "modelo.onnx"), providers=["CPUExecutionProvider"])


SESION = cargar_modelo() if os.path.exists(os.getenv("RUTA_MODELO", "modelo.onnx")) else None


def predecir(imagen):
    t0 = time.perf_counter()
    try:
        tensor, procesada = preparar_imagen(np.asarray(imagen))
    except ImagenInvalida as e:
        return str(e), "", None
    valor = float(SESION.run(None, {SESION.get_inputs()[0].name: tensor[None]})[0].squeeze()) if SESION else float("nan")
    grado = int(a_grados([valor], UMBRALES)[0]) if SESION else None
    # registro estructurado para el monitoreo (sin almacenar la imagen)
    print(json.dumps({"evento": "prediccion", "version": VERSION, "valor": valor, "grado": grado,
                      "latencia_ms": round(1000 * (time.perf_counter() - t0), 1), **estadisticas_imagen(np.asarray(imagen))}))
    return (GRADOS[grado] if grado is not None else "Modelo no cargado"), f"{valor:.2f}", procesada


demo = gr.Interface(
    fn=predecir, inputs=gr.Image(type="numpy", label="Imagen de fondo de ojo"),
    outputs=[gr.Label(label="Grado estimado"), gr.Textbox(label="Valor continuo"), gr.Image(label="Imagen procesada")],
    title="Detección de retinopatía diabética – APTOS 2019",
    description="Prototipo académico. No constituye diagnóstico médico.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
