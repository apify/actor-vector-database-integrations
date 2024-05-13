FROM apify/actor-python:3.12

WORKDIR /usr/src/app

COPY code/pyproject.toml code/poetry.lock code/pyproject.toml ./

ARG VECTOR_DATABASE
ENV POETRY_INSTALL="main${VECTOR_DATABASE:+,}${VECTOR_DATABASE}"

RUN echo "VECTOR_DATABASE: $VECTOR_DATABASE" \
    && echo "POETRY_INSTALL: $POETRY_INSTALL"

RUN echo "Python version:" \
    && python --version \
    && echo "Pip version:" \
    && pip --version \
    && echo "Installing Poetry:" \
    && pip install --no-cache-dir poetry~=1.8 \
    && echo "Installing dependencies:" \
    && poetry config virtualenvs.create false \
    && poetry install --only ${POETRY_INSTALL} --no-interaction --no-ansi \
    && rm -rf /tmp/.poetry-cache \
    && echo "All installed Python packages:" \
    && pip freeze

COPY code ./

ARG ACTOR_PATH_IN_DOCKER_CONTEXT
ENV ACTOR_PATH_IN_DOCKER_CONTEXT="${ACTOR_PATH_IN_DOCKER_CONTEXT}"

CMD ["python3", "-m", "src.entrypoint"]
