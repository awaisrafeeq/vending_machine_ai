FROM python:3.12.3
WORKDIR /app
COPY .env .
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    portaudio19-dev \
    ffmpeg \
    gcc \
    python3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD /bin/sh -c "uvicorn filehandle_databasess_copy:app --host 0.0.0.0 --port ${PORT:-8000}"