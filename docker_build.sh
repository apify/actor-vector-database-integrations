#!/usr/bin/env bash

echo "Building the Docker image"
docker build --tag vs_db --file shared/Dockerfile --build-arg ACTOR_PATH_IN_DOCKER_CONTEXT=actors/opensearch .
