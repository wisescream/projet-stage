FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY pyproject.toml README.md ./
COPY fraud_detection ./fraud_detection
COPY artifacts/model-validation.joblib /models/model-validation.joblib
COPY artifacts-replay/features.sqlite artifacts-replay/manifest.json artifacts-replay/events.jsonl artifacts-replay/labels.jsonl /bundle/
RUN pip install --no-cache-dir ".[streaming]" numpy==2.1.1 pandas==2.2.3 scikit-learn==1.5.2 joblib==1.4.2
RUN useradd --uid 10001 --create-home simulator
USER simulator
CMD ["python", "-m", "uvicorn", "fraud_detection.simulation.api:app", "--host", "0.0.0.0", "--port", "8000"]
