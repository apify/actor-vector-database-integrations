# Pinecone integration

The Apify Pinecone integration transfers selected data from Apify Actors to a Pinecone database. 
It processes the data, optionally splits it into chunks, computes embeddings, and saves them to Pinecone.

This integration supports incremental updates, updating only the data that has changed. 
This approach reduces unnecessary embedding computation and storage operations, making it suitable for search and retrieval augmented generation (RAG) use cases.

💡 **Note**: This Actor is meant to be used together with other Actors' integration sections.
For instance, if you are using the [Website Content Crawler](https://apify.com/apify/website-content-crawler), you can activate Pinecone integration to save web data as vectors to Pinecone.

For more information how to leverage vector stores in Apify platform, see [Pinecone integration](https://github.com/HonzaTuron/pinecone) and detailed blog post [what Pinecone is and why you should use it with your LLMs](https://blog.apify.com/what-is-pinecone-why-use-it-with-llms/).

## How does it work?

Apify Pinecone integration computes text embeddings and store them in Pinecone. 
It uses [LangChain](https://www.langchain.com/) to compute embeddings and interact with [Pinecone](https://www.pinecone.io/).

1. Retrieve a dataset as output from an Actor
2. [Optional] Split text data into chunks using `langchain`'s `RecursiveCharacterTextSplitter`
(enable/disable using `performChunking` and specify `chunkSize`, `chunkOverlap`)
3. Compute embeddings, e.g. using `OpenAI` or `Cohere` (specify `embeddings` and `embeddingsConfig`)
4. Update only changed data in Pinecone (enable/disable using `enableDeltaUpdates`
5. Save data into `Pinecone`

## Before you start

To utilize this integration, ensure you have:

- Created or existing `Pinecone` database. You need to know `indexName` and `apiKey`.
- An account to compute embeddings using one of the providers, e.g., OpenAI or Cohere.

## Examples

For detailed input information refer to [input schema](.actor/input_schema.json).

The configuration consists of three parts: Pinecone, embeddings provider, and data.

#### Database: Pinecone
```json
{
  "pineconeApiKey": "YOUR-PINECONE-API-KEY",
  "pineconeIndexName": "apify-pinecone"
}
```

#### Embeddings provider: OpenAI
```json 
{
  "embeddingsProvider": "OpenAIEmbeddings",
  "embeddingsApiKey": "YOUR-OPENAI-API-KEY",
  "embeddingsConfig": {"model":  "text-embedding-3-large"}
}
```

### Save data from Website Content Crawler to Pinecone

Data is transferred in the form of a dataset from [Website Content Crawler](https://apify.com/apify/website-content-crawler), which provides a dataset with the following output fields (truncated for brevity):

```json
{
  "url": "https://www.apify.com",
  "text": "Apify is a platform that enables developers to build, run, and share automation tasks.",
  "metadata": {"title": "Apify"}
}
```
This dataset is then processed by the Pinecone integration. 
In the integration settings you need to specify which fields you want to save to Pinecone, e.g., `["text"]` and which of them should be used as metadata, e.g., `{"title": "metadata.title"}`.
Without any other configuration, the data is saved to Pinecone as is.


```json
{
  "datasetFields": ["text"],
  "metadataDatasetFields": {"title": "metadata.title"}
}
```

### Create chunks from Website Content Crawler data and save them to the database

Assume that the text data from the [Website Content Crawler](https://apify.com/apify/website-content-crawler) is too long to compute embeddings. 
Therefore, we need to divide the data into smaller pieces called chunks. 
We can leverage LangChain's `RecursiveCharacterTextSplitter` to split the text into chunks and save them into a database.
The parameters `chunkSize` and `chunkOverlap` are important. 
The settings depend on your use case where a proper chunking helps optimize retrieval and ensures accurate responses.

```json
{
  "datasetFields": ["text"],
  "metadataDatasetFields": {"title": "metadata.title"},
  "performChunking": true,
  "chunkSize": 1000,
  "chunkOverlap": 0
}
```
### Incrementally update Pinecone data from Website Content Crawler

To incrementally update data from the [Website Content Crawler](https://apify.com/apify/website-content-crawler) to Pinecone, configure the integration to update only the changed or new data. 
This is controlled by the `enableDeltaUpdates` setting. 
This way, the integration minimizes unnecessary updates and ensures that only new or modified data is processed.

Further, the integration can delete data from Pinecone that hasn't been crawled for a specified period. 
It can happen that data in the Pinecone database is outdated, e.g., when a page was removed from the website. 
But it can also happen that the crawler has missed some pages due to various reasons. 
It is therefore beneficial to deleted outdated data from Pinecone database
This is controlled by the `expiredObjectDeletionPeriod` setting, where data older than the specified number of days is automatically deleted.

Concrete value of `enableDeltaUpdates` depends on your use case.
Typically, if a website is crawled every day, the `enableDeltaUpdates` can be set to 7, if you crawl every week, it can be set to 30.

```json
{
  "datasetFields": ["text"],
  "metadataDatasetFields": {"title": "metadata.title"},
  "enableDeltaUpdates": true,
  "expiredObjectDeletionPeriod": 30
}
```

### Delete outdated data from Pinecone

Furthermore, the integration can delete data from Pinecone that hasn't been crawled for a specified period. 
Outdated data can accumulate in the Pinecone database, for instance, when a page is removed from a website or when the crawler misses some pages due to various reasons.
Deleting outdated data from the Pinecone database is therefore beneficial. 
This is controlled by the `expiredObjectDeletionPeriod` setting, which automatically deletes data older than the specified number of days.

The specific value for `expiredObjectDeletionPeriod` depends on your use case.
For example, if a website is crawled daily, you might set `expiredObjectDeletionPeriod` to 7 days. 
If you crawl weekly, setting it to 30 days might be more appropriate.

Deletion of outdated data is performed only when `enableDeltaUpdates` is set to `true`.

```json
{
  "enableDeltaUpdates": true,
  "expiredObjectDeletionPeriod": 30
}
```

## Outputs

This integration will save the selected fields from your Actor to Pinecone.

## Example configuration

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