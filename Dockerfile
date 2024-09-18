FROM python:3.9-slim

WORKDIR /app

COPY darts/requirements.txt /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

RUN pip install --no-cache-dir -r requirements.txt

COPY darts/ /app/darts/

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "100", "darts.app:app"]