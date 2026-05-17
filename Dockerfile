FROM python:3.13-slim

WORKDIR /app

COPY README.md pyproject.toml uv.lock ./
COPY src ./src

RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["streamlit", "run", "src/highjump_mlops/app.py", "--server.address=0.0.0.0", "--server.port=8501"]