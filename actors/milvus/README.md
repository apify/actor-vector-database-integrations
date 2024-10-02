# Milvus integration

The Apify Milvus integration transfers selected data from Apify Actors to a  [Milvus](https://milvus.io/)/[Zilliz](https://zilliz.com) database. 
It processes the data, optionally splits it into chunks, computes embeddings, and saves them to Milvus.

This integration supports incremental updates, updating only the data that has changed. 
This approach reduces unnecessary embedding computation and storage operations, making it suitable for search and retrieval augmented generation (RAG) use cases.

💡 **Note**: This Actor is meant to be used together with other Actors' integration sections.
For instance, if you are using the [Website Content Crawler](https://apify.com/apify/website-content-crawler), you can activate Milvus integration to save web data as vectors to Milvus.

### What is Milvus/Zilliz vector database?

Milvus is an open-source vector database designed for similarity searches on large datasets of high-dimensional vectors.
Its emphasis on efficient vector similarity search enables the development of robust and scalable retrieval systems.
The Milvus database hosted at [Zilliz](https://zilliz.com/) demonstrates top performance in the [Vector Database Benchmark](https://github.com/zilliztech/VectorDBBench).

## 📋 How does the Apify-Milvus/Zilliz integration work?

Apify Milvus integration computes text embeddings and store them in Milvus. 
It uses [LangChain](https://www.langchain.com/) to compute embeddings and interact with [Milvus](https://milvus.io/).

1. Retrieve a dataset as output from an Actor
2. _[Optional]_ Split text data into chunks using `langchain`'s `RecursiveCharacterTextSplitter`
(enable/disable using `performChunking` and specify `chunkSize`, `chunkOverlap`)
3. _[Optional]_ Update only changed data in Milvus (enable/disable using `enableDeltaUpdates`)
4. Compute embeddings, e.g. using `OpenAI` or `Cohere` (specify `embeddings` and `embeddingsConfig`)
5. Save data into the database

![Apify-pinecone-integration](https://raw.githubusercontent.com/apify/actor-vector-database-integrations/master/docs/Apify-milvus-integration-readme.png)

## ✅ Before you start

To use this integration, ensure you have:

- Created or existing `Milvus` database. You need to know `milvusUri`, `milvusToken`, and `milvusCollectionName`.
- If the collection does not exist, it will be created automatically.
- An account to compute embeddings using one of the providers, e.g., [OpenAI](https://platform.openai.com/docs/guides/embeddings) or [Cohere](https://docs.cohere.com/docs/cohere-embed).

### Set up Milvus/Zilliz URI, token and collection name

You can run Milvus using Docker or try the managed Milvus service at [Zilliz](https://zilliz.com/).
For more details, please refer to the [Milvus documentation](https://milvus.io/docs).

You need the URI and Token of your Milvus/Zilliz to setup the client.
- If you have self-deployed Milvus server on [Docker or Kubernetes](https://milvus.io/docs/quickstart.md), use the server address and port as your uri, e.g.`http://localhost:19530`. If you enable the authentication feature on Milvus, use "<your_username>:<your_password>" as the token, otherwise leave the token as empty string.
- If you use [Zilliz Cloud](https://zilliz.com/cloud), the fully managed cloud service for Milvus, adjust the `uri` and `token`, which correspond to the [Public Endpoint and API key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#cluster-details) in Zilliz Cloud.

Note that the collection does not need to exist beforehand. 
It will be automatically created when data is uploaded to the database.


## 👉 Examples

The configuration consists of three parts: Milvus, embeddings provider and data.

Ensure that the vector size of your embeddings aligns with the configuration of your Milvus index. 
For instance, if you're using the `text-embedding-3-small` model from `OpenAI`, it generates vectors of size `1536`. 
This means your Milvus index should also be configured to accommodate vectors of the same size, `1536` in this case.

For detailed input information refer to the [Input page](https://apify.com/apify/milvus-integration/input-schema).

#### Database: Milvus
```json
{
  "milvusUri": "YOUR-MILVUS-URI",
  "milvusToken": "YOUR-MILVUS-TOKEN",
  "milvusCollectionName": "YOUR-MILVUS-COLLECTION-NAME"
}
```

Please refer to the instructions above on how to set up the Milvus/Zilliz `URI`, `token`, and `collection name`.

#### Embeddings provider: OpenAI
```json 
{
  "embeddingsProvider": "OpenAIEmbeddings",
  "embeddingsApiKey": "YOUR-OPENAI-API-KEY",
  "embeddingsConfig": {"model":  "text-embedding-3-large"}
}
```

### Save data from Website Content Crawler to Milvus

Data is transferred in the form of a dataset from [Website Content Crawler](https://apify.com/apify/website-content-crawler), which provides a dataset with the following output fields (truncated for brevity):

```json
{
  "url": "https://www.apify.com",
  "text": "Apify is a platform that enables developers to build, run, and share automation tasks.",
  "metadata": {"title": "Apify"}
}
```
This dataset is then processed by the Milvus integration. 
In the integration settings you need to specify which fields you want to save to Milvus, e.g., `["text"]` and which of them should be used as metadata, e.g., `{"title": "metadata.title"}`.
Without any other configuration, the data is saved to Milvus as is.


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

To incrementally update data from the [Website Content Crawler](https://apify.com/apify/website-content-crawler) to Milvus, configure the integration to update only the changed or new data. 
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

This integration will save the selected fields from your Actor to Milvus and store the chunked data in the Apify dataset.


## 🔢 Example configuration

#### Full Input Example for Website Content Crawler Actor with Milvus integration

```json
{
  "milvusUri": "YOUR-MILVUS-URI",
  "milvusToken": "YOUR-MILVUS-TOKEN",
  "milvusCollectionName": "YOUR-MILVUS-COLLECTION-NAME",
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

#### Milvus
```json
{
  "milvusUri": "YOUR-MILVUS-URI",
  "milvusToken": "YOUR-MILVUS-TOKEN",
  "milvusCollectionName": "YOUR-MILVUS-COLLECTION-NAME"
}
```

#### Managed Milvus service at [Zilliz](https://zilliz.com/)
```json
{
  "milvusUri": "https://in03-***********.api.gcp-us-west1.zillizcloud.com",
  "milvusToken": "d46**********b4b",
  "milvusCollectionName": "YOUR-MILVUS-COLLECTION-NAME"
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