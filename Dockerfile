# ===== Base =====
FROM python:3.11-slim

# Keep Python snappy & show logs immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # So imports like "from data_loader import ..." work from /app
    PYTHONPATH=/app/src

# Non-root user (good practice)
RUN useradd -ms /bin/bash appuser

# Workdir
WORKDIR /app

# OS deps (build tools help with some wheels; tini handles signals well)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential tini \
    && rm -rf /var/lib/apt/lists/*

# --- Handle your UTF-16 requirements.txt from src/ ---
# Copy just that file first for better cache behavior
COPY src/requirements.txt /tmp/requirements.txt

# Convert UTF-16 -> UTF-8 and install deps
RUN python - <<'PY'
from pathlib import Path
utf16 = Path("/tmp/requirements.txt").read_text(encoding="utf-16")
Path("/tmp/requirements-utf8.txt").write_text(utf16, encoding="utf-8")
PY
RUN pip install -r /tmp/requirements-utf8.txt

# Jupyter (your requirements doesn’t include it; add here for notebooks)
RUN pip install jupyterlab

# Copy project (code only; heavy files are ignored by .dockerignore)
COPY src ./src
COPY reports.ipynb ./reports.ipynb

# Switch to non-root
USER appuser

# Jupyter
EXPOSE 8888

# Use tini as entrypoint (clean shutdowns)
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command: run Jupyter from /app (so report.ipynb is visible)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]