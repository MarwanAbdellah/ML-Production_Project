# Two targets from one file: `api` (the assignment deliverable) and `ui`
# (Streamlit client for docker compose). The api image never installs Streamlit.
FROM python:3.14-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

RUN pip install --no-cache-dir uv==0.11.19

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH"

FROM base AS api
COPY production_app ./production_app
COPY models/model_pipeline.joblib models/selected_features.json ./models/
EXPOSE 8000
CMD ["uvicorn", "production_app.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS ui
RUN uv sync --frozen --no-dev --group ui --no-install-project
COPY streamlit_app.py ./
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
