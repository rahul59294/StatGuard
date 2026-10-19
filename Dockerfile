FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

CMD ["uv", "run", "statguard", "analyze", "samples/demo.csv", "--group", "group", "--value", "score"]
