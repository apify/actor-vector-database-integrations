# Chroma integration

The Apify Chroma integration seamlessly transfers selected data from Apify Actors to a Chroma database.

💡 **Note**: This Actor is meant to be used together with other Actors' integration sections.
For instance, if you are using the [Website Content Crawler](https://apify.com/apify/website-content-crawler), you can activate Chroma integration to save web data as vectors to Chroma.

For more information how to leverage vector stores in Apify platform, see [Pinecone integration](https://github.com/HonzaTuron/pinecone) and detailed blog post [what Pinecone is and why you should use it with your LLMs](https://blog.apify.com/what-is-pinecone-why-use-it-with-llms/).

## Description

Apify Chroma integration computes embeddings and store them in Chroma. It uses [LangChain](https://www.langchain.com/) to compute embeddings and interact with [Chroma](https://www.trychroma.io/).

1. Get `datasetId` from an `Apify Actor` output (passed automatically via integration).
2. Get dataset using `Apify Python SDK`.
3. [Optional] Split text data into chunks using `langchain`'s `RecursiveCharacterTextSplitter`
(enable/disable using `performChunking` and specify `chunkSize`, `chunkOverlap`)
4. Compute embeddings
5. Save data into `Chroma`

## Before you start

To utilize this integration, ensure you have:

- `Chroma` operational on a server or localhost.
- An account to compute embeddings using one of the providers (e.g., OpenAI or Cohere), or you can use free Huggingface models.

For quick Chroma setup, refer to [Chroma deployment](https://docs.trychroma.com/deployment#docker).
Chroma can be run in a Docker container with the following commands:

### Docker

```shell
docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma
```

### Authentication with Docker

To enable static API Token authentication, create a .env file with:

```dotenv
CHROMA_SERVER_AUTHN_CREDENTIALS=test-token
CHROMA_SERVER_AUTHN_PROVIDER=chromadb.auth.token_authn.TokenAuthenticationServerProvider
```

Then run Docker with:

```shell
docker run --env-file ./.env -p 8000:8000 chromadb/chroma
```

### If you are running Chroma locally, you can expose the localhost using Ngrok

[Install ngrok](https://ngrok.com/download) (you can use it for free or create an account). Expose Chroma using

```shell
ngrok http http://localhost:8080
```

You'll see an output similar to:
```text
Session Status                online
Account                       a@a.ai (Plan: Free)
Forwarding                    https://fdfe-82-208-25-82.ngrok-free.app -> http://localhost:8000
```

The URL (`https://fdfe-82-208-25-82.ngrok-free.app`) can be used in the as an input variable for `chromaClientHost=https://fdfe-82-208-25-82.ngrok-free.app`.
Note that your specific URL will vary.


## Inputs

For details refer to [input schema](.actor/input_schema.json).

- `chromaCollectionName`: Chroma collection name (default: `chroma`)
- `chromaClientHost`: Chroma client host
- `chromaClientPort`: Chroma client port (default: `8080`)
- `chromaClientSsl`: Enable/disable SSL (default: `false`)
- `chromaAuthCredentials`: Chroma server auth Static API token credentials
- `fields` - Array of fields you want to save. For example, if you want to push `name` and `user.description` fields, you should set this field to `["name", "user.description"]`.
- `metadataValues` - Object of metadata values you want to save. For example, if you want to push `url` and `createdAt` values to Chroma, you should set this field to `{"url": "https://www.apify.com", "createdAt": "2021-09-01"}`.
- `metadataFields` - Object of metadata fields you want to save. For example, if you want to push `url` and `createdAt` fields, you should set this field to `{"url": "url", "createdAt": "createdAt"}`. If it has the same key as `metadataValues`, it's replaced.
- `openaiApiKey` - OpenAI API KEY.
- `performChunking` - Whether to compute text chunks
- `chunkSize` - The maximum character length of each text chunk
- `chunkOverlap` - The character overlap between text chunks that are next to each other
- `embeddings` - Embeddings provider to use for computing vectors. For example, `OpenAIEmbeddings` or `CohereEmbeddings`.
- `embeddingsConfig` - Configuration for the embeddings' provider. For example, `{"model": "text-embedding-ada-002"}`.
- `embeddingsApiKey` - API key for the embeddings provider (whenever needed)

Fields `fields`, `metadataValues`, and `metadataFields` supports dot notation. For example, if you want to push `name` field from `user` object, you should set `fields` to `["user.name"]`.

## Outputs

This integration will save the selected fields from your Actor to Chroma.

## Examples

### Example input configuration (Chroma with OpenAI embeddings)

The configuration consists of three parts: Data, Chroma, and OpenAI embeddings that are typically combined, but we show them separately here for clarity.

#### Data

```json
{
  "fields": ["text"],
  "metadataValues": {"domain": "apify.com"},
  "metadataFields": {"title": "metadata.title", "loadedTime":  "crawl.loadedTime"},
  "performChunking": true,
  "chunkSize": 1000,
  "chunkOverlap": 0
}
```
#### Chroma
```json
{
  "chromaClientHost": "https://fdfe-82-208-25-82.ngrok-free.app",
  "chromaCollectionName": "chroma",
  "chromaServerAuthCredentials": "test-token"
}
```

#### OpenAI embeddings
```json 
{
  "embeddingsApiKey": "YOUR-OPENAI-API-KEY",
  "embeddings": "OpenAIEmbeddings",
  "embeddingsConfig": {"model":  "text-embedding-3-large"}
}
```
#### Cohere embeddings
```json 
{
  "embeddingsApiKey": "YOUR-COHERE-API-KEY",
  "embeddings": "CohereEmbeddings",
  "embeddingsConfig": {"model":  "embed-multilingual-v3.0"}
}
```
