FROM apify/actor-python:3.12

WORKDIR /usr/src/app

COPY code/pyproject.toml code/poetry.lock code/pyproject.toml ./

RUN echo "Python version:" \
    && python --version \
    && echo "Pip version:" \
    && pip --version \
    && echo "Installing Poetry:" \
    && pip install --no-cache-dir poetry~=1.8 \
    && echo "Installing dependencies:" \
    && poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi \
    && rm -rf /tmp/.poetry-cache \
    && echo "All installed Python packages:" \
    && pip freeze

COPY code ./

CMD ["python3", "-m", "src"]
