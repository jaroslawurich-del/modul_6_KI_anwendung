FROM python:3.11-slim

WORKDIR /app

# System dependencies (für chroma + builds)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Ordner für persistente Daten
RUN mkdir -p /app/chroma_db /app/uploads

# Streamlit Port
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]