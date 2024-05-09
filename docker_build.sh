#!/usr/bin/env bash

echo "Building the Docker image"
docker build --tag vs_db --file shared/Dockerfile --build-arg VECTOR_DATABASE=pinecone .
