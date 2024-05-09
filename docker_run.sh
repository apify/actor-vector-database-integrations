#!/usr/bin/env bash

echo "Stopping and removing the container"
docker stop vs_db 2> /dev/null || true
docker rm vs_db 2> /dev/null || true

echo "Running the container"
docker run \
    --name vs_db \
    -it \
    vs_db
