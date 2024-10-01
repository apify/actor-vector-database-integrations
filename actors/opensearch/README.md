# Amazon OpenSearch integration

The Apify Amazon Open Search integration transfers selected data from Apify Actors to a [OpenSearch](https://opensearch.org/) database. 
This integration supports [Amazon OpenSearch Service](https://aws.amazon.com/opensearch-service/) (successor of Amazon Elasticsearch Service)
and [Amazon OpenSearch Serverless](https://aws.amazon.com/opensearch-service/features/serverless/).
Also, it works with self-hosted OpenSearch instances.
The integration processes the data, optionally splits it into chunks, computes embeddings, and saves them to Open Search.

This integration supports incremental updates, updating only the data that has changed. 
This approach reduces unnecessary embedding computation and storage operations, making it suitable for search and retrieval augmented generation (RAG) use cases.

💡 **Note**: This Actor is meant to be used together with other Actors' integration sections.
For instance, if you are using the [Website Content Crawler](https://apify.com/apify/website-content-crawler), you can the integration to save web data as vectors to the database.

### What is OpenSearch database?

OpenSearch is an open-source search and analytics engine that evolved from Elasticsearch, designed to handle a variety of data types and queries, such full-text search, log analytics, and vector search). 
It supports both structured and unstructured data and is particularly useful for working with large datasets. 
OpenSearch employs inverted indices for efficient full-text searches and has integrated vector search functionalities that allow for similarity searches on high-dimensional data.
OpenSearch is also hosted as a managed service by AWS ([Amazon OpenSearch Service](https://aws.amazon.com/opensearch-service/) and [Amazon OpenSearch Serverless](https://aws.amazon.com/opensearch-service/features/serverless/)).

## 📋 How does the Apify-OpenSearch integration work?

Apify OpenSearch integration computes text embeddings and store them in the database. 
It uses [LangChain](https://www.langchain.com/) to compute embeddings and interact with the database.

1. Retrieve a dataset as output from an Actor
2. _[Optional]_ Split text data into chunks using `langchain`'s `RecursiveCharacterTextSplitter`
(enable/disable using `performChunking` and specify `chunkSize`, `chunkOverlap`)
3. _[Optional]_ Update only changed data in OpenSearch (enable/disable using `enableDeltaUpdates`)
4. Compute embeddings, e.g. using `OpenAI` or `Cohere` (specify `embeddings` and `embeddingsConfig`)
5. Save data into the database

## ✅ Before you start

To utilize this integration, ensure you have:

- Either created or have access to an existing `OpenSearch` database. You need to know several parameters, such as `openSearchUrl`, `openSearchIndexName`, and others
- An account to compute embeddings using one of the providers, e.g., [OpenAI](https://platform.openai.com/docs/guides/embeddings) or [Cohere](https://docs.cohere.com/docs/cohere-embed).

## 👉 Examples

The configuration consists of three parts: OpenSearch, embeddings provider, and data.

Ensure that the vector size of your embeddings aligns with the configuration of your index.
For instance, if you're using the `text-embedding-3-small` model from `OpenAI`, it generates vectors of size `1536`.
This means your index should also be configured to accommodate vectors of the same size, `1536` in this case.

For detailed input information refer to the [Input page](https://apify.com/apify/opensearch-integration/input-schema).

#### Database: Amazon OpenSearch Service Serverless

```json
{
  "openSearchUrl": "YOUR-OPENSEARCH-URL",
  "awsAccessKeyId": "YOUR-ACCESS-KEY-ID",
  "awsSecretAccessKey": "YOUR-SECRET-ACCESS-KEY",
  "openSearchIndexName": "YOUR-OPENSEARCH-INDEX-NAME",
  "autoCreateIndex": true
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

### Save data from Website Content Crawler to OpenSearch

Data is transferred in the form of a dataset from [Website Content Crawler](https://apify.com/apify/website-content-crawler), which provides a dataset with the following output fields (truncated for brevity):

```json
{
  "url": "https://www.apify.com",
  "text": "Apify is a platform that enables developers to build, run, and share automation tasks.",
  "metadata": {"title": "Apify"}
}
```
This dataset is then processed by the OpenSearch integration. 
In the integration settings you need to specify which fields you want to save to OpenSearch, e.g., `["text"]` and which of them should be used as metadata, e.g., `{"title": "metadata.title"}`.
Without any other configuration, the data is saved to OpenSearch as is.


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

To incrementally update data from the [Website Content Crawler](https://apify.com/apify/website-content-crawler) to database, configure the integration to update only the changed or new data. 
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


## 💾 Outputs

This integration will save the selected fields from your Actor to database and store the chunked data in the Apify dataset.


## 🔢 Example configuration

#### Full Input Example for Website Content Crawler Actor with Amazon OpenSearch integration

```json
{
  "openSearchUrl": "YOUR-OPENSEARCH-URL",
  "awsAccessKeyId": "YOUR-ACCESS-KEY-ID",
  "awsSecretAccessKey": "YOUR-SECRET-ACCESS-KEY",
  "openSearchIndexName": "YOUR-OPENSEARCH-INDEX-NAME",
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

#### Database: Amazon OpenSearch Service
```json
{
  "openSearchUrl": "YOUR-OPENSEARCH-URL",
  "awsAccessKeyId": "YOUR-ACCESS-KEY-ID",
  "awsSecretAccessKey": "YOUR-SECRET-ACCESS-KEY",
  "openSearchIndexName": "YOUR-OPENSEARCH-INDEX-NAME",
  "awsServiceName": "es"
}
```

#### Database: Self-hosted OpenSearch
```json
{
  "openSearchUrl": "YOUR-OPENSEARCH-URL",
  "openSearchIndexName": "YOUR-OPENSEARCH-INDEX-NAME",
  "useAwsV4Auth": false,
  "useSsl": false,
  "verifyCerts": false
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