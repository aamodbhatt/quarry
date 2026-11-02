# quarry: the agent, CLI, web UI and MCP server in one image.
# Build:  docker build -t quarry .
# Run:    docker run --rm -p 127.0.0.1:8097:8097 --env-file .env -v quarry-data:/app/data quarry
#         (add `serve --offline` to run the scripted demo without a key)

FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 10001 quarry
WORKDIR /app
COPY --from=build --chown=quarry:quarry /app /app
COPY --chown=quarry:quarry configs ./configs
COPY --chown=quarry:quarry docker/entrypoint.sh /usr/local/bin/quarry-entrypoint
RUN chmod 0755 /usr/local/bin/quarry-entrypoint && mkdir -p /app/data /app/runs && chown quarry:quarry /app/data /app/runs
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER quarry
EXPOSE 8097
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8097/api/info', timeout=4)"
ENTRYPOINT ["quarry-entrypoint"]
CMD ["serve", "--host", "0.0.0.0"]
