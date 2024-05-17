FROM apify/actor-python:3.11


WORKDIR /usr/src/app

COPY code/pyproject.toml code/poetry.lock code/pyproject.toml ./

ARG ACTOR_PATH_IN_DOCKER_CONTEXT
ENV ACTOR_PATH_IN_DOCKER_CONTEXT="${ACTOR_PATH_IN_DOCKER_CONTEXT}"

RUN echo "Python version:" \
    && python --version \
    && echo "Pip version:" \
    && pip --version \
    && echo "Installing Poetry:" \
    && pip install --no-cache-dir poetry~=1.8 \
    && echo "Installing dependencies:" \
    && poetry config virtualenvs.create false \
    && poetry install --only "main,${ACTOR_PATH_IN_DOCKER_CONTEXT#actors/}" --no-interaction --no-ansi \
    && rm -rf /tmp/.poetry-cache \
    && echo "All installed Python packages:" \
    && pip freeze

COPY code ./

CMD ["python3", "-m", "src"]
