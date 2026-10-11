FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements-ci.txt .
RUN pip install --no-cache-dir -r requirements-ci.txt onnxruntime gradio
COPY src/ src/
COPY app/ app/
ENV PYTHONPATH=/app/src
EXPOSE 7860
CMD ["python", "app/app.py"]
