# GENIE Learn Prototype — Dockerfile
# ===================================
# docker build -t genie-prototype .
# docker run -p 8501:8501 --env-file .env genie-prototype
#
# Sin .env funciona en modo demo (mock LLM, TF-IDF retrieval).
# Con OPENAI_API_KEY en .env: embeddings reales + GPT-4o-mini.

FROM python:3.11-slim

WORKDIR /app

# Dependencias de sistema (para PyMuPDF si se necesita)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Puerto de Streamlit
EXPOSE 8501

# Healthcheck — útil si esto acaba en un cluster
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Streamlit necesita estas configs para funcionar en Docker
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# No preguntar por telemetría
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app.py"]
