# Pinecone

The Apify Pinecone integration seamlessly transfers selected data from Apify Actors to a Pinecone database.

💡 **Note**: This Actor is meant to be used together with other Actors' integration sections.
For instance, if you are using the [Website Content Crawler](https://apify.com/apify/website-content-crawler), you can activate Pinecone integration to save web data as vectors to Pinecone.

For more information how to leverage vector stores in Apify platform, see [Pinecone integration](https://github.com/HonzaTuron/pinecone) and detailed blog post [what Pinecone is and why you should use it with your LLMs](https://blog.apify.com/what-is-pinecone-why-use-it-with-llms/).

## Description

Apify Pinecone integration computes embeddings and store them in Pinecone. It uses [LangChain](https://www.langchain.com/) to compute embeddings and interact with [Pinecone](https://www.pinecone.io/).

1. Get `datasetId` from an `Apify Actor` output (passed automatically via integration).
2. Get dataset using `Apify Python SDK`.
3. [Optional] Split text data into chunks using `langchain`'s `RecursiveCharacterTextSplitter`
(enable/disable using `performChunking` and specify `chunkSize`, `chunkOverlap`)
4. Compute embeddings
5. Save data into `Pinecone`

## Before you start

To utilize this integration, ensure you have:

- Created or existing `Pinecone` database. You need to know `indexName` and `apiKey`.
- An account to compute embeddings using one of the providers, e.g., OpenAI or Cohere.

## Inputs

For details refer to [input schema](.actor/input_schema.json).

- `pineconeIndexName`: Pinecone index name
- `pineconeApiKey`: Pinecone API key
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

This integration will save the selected fields from your Actor to Pinecone.

## Examples

### Example input configuration (Pinecone with OpenAI embeddings)

The configuration consists of three parts: Data, Pinecone, and OpenAI embeddings that are typically combined, but we show them separately here for clarity.

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
#### Pinecone
```json
{
  "pineconeApiKey": "YOUR-PINECONE-API-KEY",
  "pineconeIndexName": "apify-pinecone"
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