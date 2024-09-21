FROM python:3.9-slim

WORKDIR /app

COPY darts/ /app/darts/
COPY darts/requirements.txt /app/

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "100", "darts.app:app"]