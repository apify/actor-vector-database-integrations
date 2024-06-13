# Qdrant integration

The Apify Qdrant integration transfers selected data from Apify Actors to a [Qdrant](https://qdrant.tech/) collection.
It processes the data, optionally splits it into chunks, computes embeddings, and saves them to a Qdrant collection.

This integration supports incremental updates, updating only the data that has changed.
This approach reduces unnecessary embedding computation and storage operations, making it suitable for search and retrieval augmented generation (RAG) use cases.

💡 **Note**: This Actor is meant to be used together with other Actors' integration sections.
For instance, if you are using the [Website Content Crawler](https://apify.com/apify/website-content-crawler), you can activate Qdrant integration to save web data as vectors in a Qdrant collection.

## How does it work?

It uses [LangChain](https://www.langchain.com/) to compute vector embeddings and interact with [Qdrant](https://www.qdrant.tech/).

1. Retrieve a dataset as output from an Actor
2. [Optional] Split text data into chunks using `langchain`'s `RecursiveCharacterTextSplitter`
(enable/disable using `performChunking` and specify `chunkSize`, `chunkOverlap`)
3. [Optional] Update only changed data in Qdrant (enable/disable using `enableDeltaUpdates`)
4. Compute embeddings, e.g. using `OpenAI` or `Cohere` (specify `embeddings` and `embeddingsConfig`)
5. Save data into `Qdrant`

## Before you start

To utilize this integration, ensure you have:

- A Qdrant instance to connect to. You can get a free cloud instance at [cloud.qdrant.io](https://cloud.qdrant.io/).
- An account to compute embeddings using one of the providers, e.g., OpenAI or Cohere.

## Examples

For detailed input information refer to [input schema](.actor/input_schema.json).

The configuration consists of three parts: Qdrant, embeddings provider, and data.

#### Database: Qdrant

```json
{
  "qdrantUrl": "https://xyz-example.eu-central.aws.cloud.qdrant.io",
  "qdrantApiKey": "YOUR-QDRANT-API-KEY",
  "qdrantCollectionName": "apify-qdrant"
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

### Save data from Website Content Crawler to Qdrant

Data is transferred in the form of a dataset from [Website Content Crawler](https://apify.com/apify/website-content-crawler), which provides a dataset with the following output fields (truncated for brevity):

```json
{
  "url": "https://www.apify.com",
  "text": "Apify is a platform that enables developers to build, run, and share automation tasks.",
  "metadata": {"title": "Apify"}
}
```

This dataset is then processed by the Qdrant integration.
In the integration settings you need to specify which fields you want to save to Qdrant, e.g., `["text"]` and which of them should be used as metadata, e.g., `{"title": "metadata.title"}`.
Without any other configuration, the data is saved to Qdrant as is.

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

### Incrementally update database from the Website Content Crawler

To incrementally update data from the [Website Content Crawler](https://apify.com/apify/website-content-crawler) to Qdrant, configure the integration to update only the changed or new data.
This is controlled by the `enableDeltaUpdates` setting.
This way, the integration minimizes unnecessary updates and ensures that only new or modified data is processed.

A checksum is computed for each dataset item (together with all metadata) and stored in the database alongside the vectors.
When the data is re-crawled, the checksum is recomputed and compared with the stored checksum.
If the checksum is different, the old data (including vectors) is deleted and new data is saved.
Otherwise, only the `last_seen_at` metadata field is updated to indicate when the data was last seen.

#### Provide unique identifier for each dataset item

To incrementally update the data, you need to be able to uniquely identify each dataset item.
The variable `deltaUpdatesPrimaryDatasetFields` specifies which fields are used to uniquely identify each dataset item and helps track content changes across different crawls.
For instance, when working with the Website Content Crawler, you can use the URL as a unique identifier.

```json
{
  "enableDeltaUpdates": true,
  "deltaUpdatesPrimaryDatasetFields": ["url"]
}
```

#### Delete outdated data

The integration can also delete data from Qdrant that hasn't been crawled for a specified period.
This is useful when data in the Qdrant collection becomes outdated, such as when a page is removed from a website.

This is controlled by the `expiredObjectDeletionPeriodDays` setting, which automatically deletes data older than the specified number of days.
For each crawl, the `last_seen_at` metadata field is updated.
When a database object has not been seen for more than `expiredObjectDeletionPeriodDays` days, it is deleted.

The specific value of `expiredObjectDeletionPeriodDays` depends on your use case.
Typically, if a website is crawled daily, `expiredObjectDeletionPeriodDays` can be set to 7.
If you crawl weekly, it can be set to 30.
To disable this feature, set `expiredObjectDeletionPeriodDays` to `0`.

```json
{
  "enableDeltaUpdates": true,
  "expiredObjectDeletionPeriodDays": 30
}
```

## Outputs

This integration will save the selected fields from your Actor to Qdrant.

## Example configuration

#### Full Input Example for Website Content Crawler Actor with Qdrant integration

```json
{
  "qdrantApiKey": "YOUR-QDRANT-API-KEY",
  "qdrantUrl": "https://xyz-example.eu-central.aws.cloud.qdrant.io",
  "qdrantCollectionName": "apify-qdrant",
  "embeddingsApiKey": "YOUR-OPENAI-API-KEY",
  "embeddingsConfig": {
    "model": "text-embedding-3-small"
  },
  "embeddingsProvider": "OpenAI",
  "datasetFields": [
    "text"
  ],
  "enableDeltaUpdates": true,
  "deltaUpdatesPrimaryDatasetFields": ["url"],
  "expiredObjectDeletionPeriodDays": 7,
  "performChunking": true,
  "chunkSize": 2000,
  "chunkOverlap": 200
}
```

#### Qdrant

```json
{
  "qdrantUrl": "https://xyz-example.eu-central.aws.cloud.qdrant.io",
  "qdrantApiKey": "YOUR-QDRANT-API-KEY",
  "qdrantCollectionName": "apify-qdrant",
  "qdrantAutoCreateCollection": true
}
```

#### OpenAI embeddings

```json
{
  "embeddingsApiKey": "YOUR-OPENAI-API-KEY",
  "embeddings": "OpenAI",
  "embeddingsConfig": {"model":  "text-embedding-3-large"}
}
```

#### Cohere embeddings

```json
{
  "embeddingsApiKey": "YOUR-COHERE-API-KEY",
  "embeddings": "Cohere",
  "embeddingsConfig": {"model":  "embed-multilingual-v3.0"}
}
```
