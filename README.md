# Vector Database Integrations

This repository contains Actors for different vector databases. 
The Actors are designed to be used with other Apify Actors as an integration.

## Features
- Fetch data from other Apify Actors
- Chunk data into smaller pieces (use tokens not characters)
- Compute embeddings (OpenAI, HuggingFace, Cohere) with different models
- Store vectors in vector databases (Pinecone, Chroma)

## Features - Roadmap
- Add other state-of-the-art embeddings (e.g., HuggingFace BGE, HuggingFace inference endpoint)
- Embeddings - use tokens for chunking (not characters)
- Add option to drop all existing vectors in a database (not supported by LangChain, custom implementation needed)
- Add option to replace vectors with a filter (e.g., replace only vectors with a specific tag, URL, etc.). Not supported by LangChain, custom implementation needed.
- Add PGVector, Weaviate


## Code
Contains the core functionality for all the vector databases and integrations.

## Supported Vector Embeddings
- OpenAI
- Cohere

## Supported Vector Databases
- Pinecone
- Chroma
