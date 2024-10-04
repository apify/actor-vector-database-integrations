# Change Log

## 0.1.6 (2024-10-04)

- Add retry logic with exponential backoff for Qdrant errors.
- Use Langchain Fake Embeddings for testing.
- Parallelize comparison of crawled data with the database.
- Implement a count() method for Pinecone and Qdrant.

## 0.1.5 (2024-09-30)

- Add retry logic with exponential backoff for Pinecone database to handle rate limiting.

## 0.1.4 (2024-09-24)

- Change `milvusUrl` to `milvusUri` and `milvusApiKey` to `milvusToken`. 
- Remove `milvusUsername` and `milvusPassword` as they were moved to the `milvusUri` as part of the URI.

## 0.1.3 (2024-09-11)

- Set `performChunking` to `true` by default as some users were not aware of the setting their runs were failing.

## 0.1.2 (2024-09-04)

- Pinecone - add support to update Pinecone index namespace (Optional arg: `PineconeIndexNamespace`)

## 0.1.1 (2024-07-24)

- Fixed issue with Weaviate database when collection was not created before inserting data.

## 0.1.0 (2024-07-24)

- Introduced the `deleteExpiredObjects` setting to enable or disable the automatic deletion feature.
- Previously, outdated data deletion was tied to the enableDeltaUpdates setting. Now, the deletion feature can be controlled independently.

## 0.0.4 (2024-07-12)

- Add integration to Milvus database (Zilliz)

## 0.0.3 (2024-06-26)

- The key in the metadata dictionary has been refactored from 'id' to 'chunk_id'. This change was necessitated by the requirements of the Weaviate database.
- A new function, 'get_by_item_id', has been introduced. This function fetches objects from the database, including their ids. With this change, it's no longer necessary to store the 'chunk_id' in the metadata. However, storing the 'chunk_id' remains beneficial for referencing chunks within the metadata."

## 0.0.2 (2024-06-24)
- Add PostgreSQL database integration (PGVector integration)

## 0.0.1 (2024-06-20)
- Improved README.md and error messages
 
## 0.0.0 (2024-05-24)
- Initial import
