FROM python:3.14.3-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data

EXPOSE 8000
EXPOSE 8501

#Lancement local
#CMD ["uvicorn", "mlops_tp.api:app", "--host", "0.0.0.0", "--port", "8000"]

# - utilisation de la variable d'environnement PORT, si PORT n'est pas défini, on utilise 8000 par défaut
# - l'API écoute toujours sur (toutes les adresses) 0.0.0.0 pour être accessible depuis l'extérieur
CMD ["sh", "-c", "uvicorn mlops_tp.api:app --host 0.0.0.0 --port ${PORT:-8000}"]