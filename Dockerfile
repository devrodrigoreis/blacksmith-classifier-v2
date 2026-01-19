#FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
#RUN pip install fastapi uvicorn transformers pandas scikit-learn joblib pyyaml

FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .

# Explicitly uninstall numpy (in case it's partially installed)
RUN pip uninstall -y numpy

# Install numpy first
RUN pip install --no-cache-dir numpy

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Crie o diretório 'models'
RUN mkdir models
COPY models/ models/

COPY models/bert_category_encoder.joblib /app/models/
COPY fallback_tokenizer.joblib /app/
COPY fallback_label_encoder.joblib /app/
COPY data/categories.csv /app/data/
COPY api/main.py /app/api/
COPY api/config.yaml /app/api/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]