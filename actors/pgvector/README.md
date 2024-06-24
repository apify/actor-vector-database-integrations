# PGVector integration

The Apify PGVector integration transfers selected data from Apify Actors to a PGVector database. 
It processes the data, optionally splits it into chunks, computes embeddings, and saves them to PGVector.

This integration supports incremental updates, updating only the data that has changed. 
This approach reduces unnecessary embedding computation and storage operations, making it suitable for search and retrieval augmented generation (RAG) use cases.

💡 **Note**: This Actor is meant to be used together with other Actors' integration sections.
For instance, if you are using the [Website Content Crawler](https://apify.com/apify/website-content-crawler), you can activate PGVector integration to save web data as vectors to PGVector.

## How does it work?

Apify PGVector integration computes text embeddings and store them in PGVector. 
It uses [LangChain](https://www.langchain.com/) to compute embeddings and interact with [PGVector](https://github.com/pgvector/pgvector).

1. Retrieve a dataset as output from an Actor
2. _[Optional]_ Split text data into chunks using `langchain`'s `RecursiveCharacterTextSplitter`
(enable/disable using `performChunking` and specify `chunkSize`, `chunkOverlap`)
3. _[Optional]_ Update only changed data in PGVector (enable/disable using `enableDeltaUpdates`)
4. Compute embeddings, e.g. using `OpenAI` or `Cohere` (specify `embeddings` and `embeddingsConfig`)
5. Save data into the database

## Before you start

To utilize this integration, ensure you have:

- Created or existing `PGVector` database. You need to know `indexName` and `apiKey`.
- An account to compute embeddings using one of the providers, e.g., OpenAI or Cohere.

## Examples

For detailed input information refer to [input schema](.actor/input_schema.json).

The configuration consists of three parts: PGVector, embeddings provider, and data.

Ensure that the vector size of your embeddings aligns with the configuration of your PGVector index. 
For instance, if you're using the `text-embedding-3-small` model from `OpenAI`, it generates vectors of size `1536`. 
This means your PGVector index should also be configured to accommodate vectors of the same size, `1536` in this case.

#### Database: PGVector
```json
{
  "postgresSqlConnectionStr": "postgresql://postgres:password@localhost:5432/apify",
  "postgresCollectionName": "apify-collection"
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

### Save data from Website Content Crawler to PGVector

Data is transferred in the form of a dataset from [Website Content Crawler](https://apify.com/apify/website-content-crawler), which provides a dataset with the following output fields (truncated for brevity):

```json
{
  "url": "https://www.apify.com",
  "text": "Apify is a platform that enables developers to build, run, and share automation tasks.",
  "metadata": {"title": "Apify"}
}
```
This dataset is then processed by the PGVector integration. 
In the integration settings you need to specify which fields you want to save to PGVector, e.g., `["text"]` and which of them should be used as metadata, e.g., `{"title": "metadata.title"}`.
Without any other configuration, the data is saved to PGVector as is.


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

To incrementally update data from the [Website Content Crawler](https://apify.com/apify/website-content-crawler) to PGVector, configure the integration to update only the changed or new data. 
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

The integration can also delete data from PGVector that hasn't been crawled for a specified period. 
This is useful when data in the PGVector database becomes outdated, such as when a page is removed from a website. 

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

This integration will save the selected fields from your Actor to PGVector.

## Example configuration

#### Full Input Example for Website Content Crawler Actor with PGVector integration

```json
{
  "postgresSqlConnectionStr": "postgresql://postgres:password@localhost:5432/apify",
  "postgresCollectionName": "apify-collection",
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

#### PGVector
```json
{
  "postgresSqlConnectionStr": "postgresql://postgres:password@localhost:5432/apify",
  "postgresCollectionName": "apify-collection"
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
### Local postgres database with PGVector
To start a local Postgres database with PGVector using Docker, refer to the docker-compose.yaml file and run the following command:

```bash
docker-compose up
```
You can connect to the database using psql with the following command:

```bash
psql -h localhost -p 5324 -U postgres -d apify
```
LangChain uses the concept of collections to store data. 
Collections help to separate data for different projects or use cases. 
For each collection, two tables are created: `langchain_pg_embedding` and `langchain_pg_collection`. 
The `langchain_pg_embedding` table stores the embeddings, page_content, and associated metadata. 
The `langchain_pg_collection` table stores the list of collections. 
LangChain will automatically create these tables when the first embedding is saved to a collection.