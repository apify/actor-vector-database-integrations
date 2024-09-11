# Pinecone integration

The Apify Pinecone integration transfers selected data from Apify Actors to a [Pinecone](https://www.pinecone.io) database. 
It processes the data, optionally splits it into chunks, computes embeddings, and saves them to Pinecone.

This integration supports incremental updates, updating only the data that has changed. 
This approach reduces unnecessary embedding computation and storage operations, making it suitable for search and retrieval augmented generation (RAG) use cases.

💡 **Note**: This Actor is meant to be used together with other Actors' integration sections.
For instance, if you are using the [Website Content Crawler](https://apify.com/apify/website-content-crawler), you can activate Pinecone integration to save web data as vectors to Pinecone.

For more information how to leverage vector stores in Apify platform, see detailed blog post [What Pinecone is and why you should use it with your LLMs](https://blog.apify.com/what-is-pinecone-why-use-it-with-llms/).
For setup instructions in the Apify Console, see the [Pinecone integration guide](https://docs.apify.com/platform/integrations/pinecone).

### What is Pinecone vector database?

[Pinecone](https://www.pinecone.io/) is a cloud-native vector database designed to manage high-dimensional vector data efficiently. 
Its core functionality is based on the Approximate Nearest Neighbor (ANN) search, which quickly identifies and ranks matches within large datasets. 
The serverless architecture of Pinecone enables rapid development and deployment of powerful applications. 
Pinecone supports clients in Python, Java, Node.js, and Golang.

## 📋 How does the Apify-Pinecone integration work?

Apify Pinecone integration computes text embeddings and store them in Pinecone. 
It uses [LangChain](https://www.langchain.com/) to compute embeddings and interact with [Pinecone](https://www.pinecone.io/).

1. Retrieve a dataset as output from an Actor
2. _[Optional]_ Split text data into chunks using `langchain`'s `RecursiveCharacterTextSplitter`
(enable/disable using `performChunking` and specify `chunkSize`, `chunkOverlap`)
3. _[Optional]_ Update only changed data in Pinecone (enable/disable using `enableDeltaUpdates`)
4. Compute embeddings, e.g. using `OpenAI` or `Cohere` (specify `embeddings` and `embeddingsConfig`)
5. Save data into the database

![Apify-pinecone-integration](https://github.com/apify/actor-vector-database-integrations/blob/master/docs/Apify-pinecone-integration-readme.png)

## ✅ Before you start

To utilize this integration, ensure you have:

- Created or existing `Pinecone` database. You need to know `indexName` and `apiKey`.
- An account to compute embeddings using one of the providers, e.g., OpenAI or Cohere.

## 👉 Examples

The configuration consists of three parts: Pinecone, embeddings provider, and data.

Ensure that the vector size of your embeddings aligns with the configuration of your Pinecone index. 
For instance, if you're using the `text-embedding-3-small` model from `OpenAI`, it generates vectors of size `1536`.
This means your Pinecone index should also be configured to accommodate vectors of the same size, `1536` in this case.

For detailed input information refer to the [Input page](./input-schema).

#### Database: Pinecone
```json
{
  "pineconeApiKey": "YOUR-PINECONE-API-KEY",
  "pineconeIndexName": "apify"
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
### Incrementally update database from the Website Content Crawler

To incrementally update data from the [Website Content Crawler](https://apify.com/apify/website-content-crawler) to Pinecone, configure the integration to update only the changed or new data. 
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

This integration will save the selected fields from your Actor to Pinecone and store the chunked data in the Apify dataset.


## 🔢 Example configuration

#### Full Input Example for Website Content Crawler Actor with Pinecone integration

```json
{
  "pineconeApiKey": "YOUR-PINECONE-API-KEY",
  "pineconeIndexName": "apify",
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

#### Pinecone
```json
{
  "pineconeApiKey": "YOUR-PINECONE-API-KEY",
  "pineconeIndexName": "apify"
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