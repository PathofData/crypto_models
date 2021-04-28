FROM python:3.6-slim

RUN pip install --upgrade pip

RUN mkdir /app

WORKDIR /app

RUN mkdir /output

VOLUME /output

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN mkdir /app/saved_models

COPY saved_models/*.joblib /app/saved_models/

COPY crypto_modules.py crypto_predictions.py feature_generation.py /app/

ENV BASE_DIR /app

CMD ["python", "/app/crypto_predictions.py"]