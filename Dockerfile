# Matches the Python version the local .venv was built against (3.13).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install CPU-only torch BEFORE requirements.txt. sentence-transformers depends
# on torch, and on Linux pip would otherwise resolve the default CUDA build -
# several GB of GPU runtime this image can never use. Pinning the CPU wheel
# index first means the later resolve sees torch already satisfied.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copied separately so dependency layers stay cached when only src/ changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY food_recipes.csv .

# 7860 = Gradio UI, 8000 = FastAPI. Publishing is decided in compose.
EXPOSE 7860 8000

# Bind to all interfaces: on 127.0.0.1 the published port is unreachable from
# the host. No share tunnel by default for a locally-run container.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SHARE=false

CMD ["python", "-m", "src.gradio_app"]
