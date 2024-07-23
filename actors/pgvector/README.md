# PostgreSQL (with PGVector) integration

The Apify PGVector integration transfers selected data from Apify Actors to PostgreSQL (with PGVector extension). 
It processes the data, optionally splits it into chunks, computes embeddings, and saves them to PostgreSQL.

This integration supports incremental updates, updating only the data that has changed. 
This approach reduces unnecessary embedding computation and storage operations, making it suitable for search and retrieval augmented generation (RAG) use cases.

💡 **Note**: This Actor is meant to be used together with other Actors' integration sections.
For instance, if you are using the [Website Content Crawler](https://apify.com/apify/website-content-crawler), you can activate PGVector integration to save web data as vectors to PostgreSQL.

## How does it work?

Apify PGVector integration computes text embeddings and store them in PostgreSQL. 
It uses [LangChain](https://www.langchain.com/) to compute embeddings and interact with [PGVector](https://github.com/pgvector/pgvector).

1. Retrieve a dataset as output from an Actor
2. _[Optional]_ Split text data into chunks using `langchain`'s `RecursiveCharacterTextSplitter`
(enable/disable using `performChunking` and specify `chunkSize`, `chunkOverlap`)
3. _[Optional]_ Update only changed data in PostgreSQL (enable/disable using `enableDeltaUpdates`)
4. Compute embeddings, e.g. using `OpenAI` or `Cohere` (specify `embeddings` and `embeddingsConfig`)
5. Save data into the database

## Before you start

To utilize this integration, ensure you have:

- Created or existing `PostgreSQL` database with PGVector extension. You need to know `postgresSqlConnectionStr` and `postgresCollectionName`.
- An account to compute embeddings using one of the providers, e.g., [OpenAI](https://platform.openai.com/docs/guides/embeddings) or [Cohere](https://docs.cohere.com/docs/cohere-embed).

## Examples

For detailed input information refer to [input schema](.actor/input_schema.json).

The configuration consists of three parts: PGVector, embeddings provider, and data.

Ensure that the vector size of your embeddings aligns with the configuration of your PostgreSQL. 
For instance, if you're using the `text-embedding-3-small` model from `OpenAI`, it generates vectors of size `1536`. 
This means your PostgreSQL vector should also be configured to accommodate vectors of the same size, `1536` in this case.

#### Database: PostgreSQL with PGVector
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

### Save data from Website Content Crawler to PostgreSQL

Data is transferred in the form of a dataset from [Website Content Crawler](https://apify.com/apify/website-content-crawler), which provides a dataset with the following output fields (truncated for brevity):

```json
{
  "url": "https://www.apify.com",
  "text": "Apify is a platform that enables developers to build, run, and share automation tasks.",
  "metadata": {"title": "Apify"}
}
```
This dataset is then processed by the PGVector integration. 
In the integration settings you need to specify which fields you want to save to PostgreSQL, e.g., `["text"]` and which of them should be used as metadata, e.g., `{"title": "metadata.title"}`.
Without any other configuration, the data is saved to PostgreSQL as is.


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

To incrementally update data from the [Website Content Crawler](https://apify.com/apify/website-content-crawler) to PostgreSQL, configure the integration to update only the changed or new data. 
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

### Delete outdated (expired) data

The integration can delete data from the database that hasn't been crawled for a specified period, which is useful when data becomes outdated, such as when a page is removed from a website.

The deletion feature can be enabled or disabled using the `deleteExpiredObjects` setting.

For each crawl, the `last_seen_at` metadata field is created or updated.
This field records the most recent time the data object was crawled.
The `expiredObjectDeletionPeriodDays` setting is used to control number of days since the last crawl, after which the data object is considered expired.
If a database object has not been seen for more than the `expiredObjectDeletionPeriodDays`, it will be deleted automatically.

The specific value of `expiredObjectDeletionPeriodDays` depends on your use case. 
- If a website is crawled daily, `expiredObjectDeletionPeriodDays` can be set to 7. 
- If you crawl weekly, it can be set to 30.

To disable this feature, set `deleteExpiredObjects` to `false`.

```json
{
  "deleteExpiredObjects": true,
  "expiredObjectDeletionPeriodDays": 30
}
```

💡 If you are using multiple Actors to update the same database, ensure that all Actors crawl the data at the same frequency. 
Otherwise, data crawled by one Actor might expire due to inconsistent crawling schedules.


## Outputs

This integration will save the selected fields from your Actor to PostgreSQL.

## Example configuration

#### Full Input Example for Website Content Crawler Actor with PostgreSQL integration

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

#### PostgreSQL with PGVector
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
### Local postgres database with PostgreSQL with PGVector
To start a local PostgresSQL database with PGVector using Docker, refer to the docker-compose.yaml file and run the following command:

```bash
docker-compose up
```
You can connect to the database using psql with the following command:

```bash
psql -h localhost -p 5324 -U postgres -d apify
```

Or you can use [PGAdmin](https://www.pgadmin.org/) to connect to the database.
```bash
docker run -e PGADMIN_DEFAULT_EMAIL=*@apify.com -e PGADMIN_DEFAULT_PASSWORD=root -p 8000:80 dpage/pgadmin4
```

LangChain uses the concept of collections to store data. 
Collections help to separate data for different projects or use cases. 
For each collection, two tables are created: `langchain_pg_embedding` and `langchain_pg_collection`. 
The `langchain_pg_embedding` table stores the embeddings, page_content, and associated metadata. 
The `langchain_pg_collection` table stores the list of collections. 
LangChain will automatically create these tables when the first embedding is saved to a collection.