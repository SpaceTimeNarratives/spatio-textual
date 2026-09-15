FROM python:3.11-slim

WORKDIR /app
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements-lite.txt ./
COPY spatio_textual ./spatio_textual
COPY app.py ./app.py
COPY example-texts ./example-texts
COPY tutorials ./tutorials

RUN python -m pip install --upgrade pip wheel && \
    python -m pip install -r requirements-lite.txt

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
