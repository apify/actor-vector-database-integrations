# ChromaDB integration

The Apify ChromaDB integration seamlessly transfers selected data from Apify Actors to a ChromaDB.

Is there anything you find unclear or missing? Please don't hesitate to inform us by creating an issue.

⚠️ **Note**: This Actor is meant to be used together with other Actors' integration section.
For instance, if you are using the [Website Content Crawler](https://apify.com/apify/website-content-crawler),
you can enable ChromaDB integration to store vectors in ChromaDB.

Apify ChromaDB integration computes OpenAI embeddings and store them in ChromaDB. It uses [LangChain](https://www.langchain.com/)
to interact with [OpenAI embeddings](https://platform.openai.com/docs/guides/embeddings) and [ChromaDB](https://www.trychroma.com/).

For more information how to leverage vector stores in Apify platform, see a similar [Pinecone
integration](https://github.com/HonzaTuron/pinecone) and detailed blog post [what Pinecone is and why you should use it with your LLMs](https://blog.apify.com/what-is-pinecone-why-use-it-with-llms/).

## Description

The ChromaDB-integration is designed to compute and store vectors from other Actors' data. It uses langchain
to interact with `OpenAI` and `ChromaDB`.

1. Get `dataset_id` from an `Apify Actor` output (passed automatically via integration).
2. Get dataset using `Apify Python SDK`.
3. [Optional] Split text data into chunks using `langchain`'s `RecursiveCharacterTextSplitter`
(enable/disable using `perform_chunking` and specify `chunk_size`, `chunk_overlap`)
4. Compute embeddings using `OpenAI`
5. Save data into `ChromaDB`

## Before you start

To utilize this integration, ensure you have:

- An OpenAI account and an OpenAI API token. Create a free account at [OpenAI](https://beta.openai.com/).
- `ChromaDB` operational on a server or localhost.

For quick ChromaDB setup, refer to [ChromaDB deployment](https://docs.trychroma.com/deployment#docker).
ChromaDB can be run in a Docker container with the following commands:

### Docker

```shell
docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma
```

### Authentication with Docker

To enable static API Token authentication, create a .env file with:

```dotenv
CHROMA_SERVER_AUTH_CREDENTIALS=test-token
CHROMA_SERVER_AUTH_CREDENTIALS_PROVIDER=chromadb.auth.token.TokenConfigServerAuthCredentialsProvider
CHROMA_SERVER_AUTH_PROVIDER=chromadb.auth.token.TokenAuthServerProvider
```

Then run Docker with:

```shell
docker run --env-file ./.env -p 8000:8000 chromadb/chroma
```

### If you are running ChromaDB locally, you can expose the localhost using Ngrok

[Install ngrok](https://ngrok.com/download) (you can use it for free or create an account). Expose ChromaDB using

```shell
ngrok http http://localhost:8080
```

You'll see an output similar to:
```text
Session Status                online
Account                       a@a.ai (Plan: Free)
Forwarding                    https://fdfe-82-208-25-82.ngrok-free.app -> http://localhost:8000
```

The URL (`https://fdfe-82-208-25-82.ngrok-free.app`) can be used in the as an input variable for `chroma_client_host`.
Note that your specific URL will vary.


## Inputs

For details refer to [input schema](.actor/input_schema.json).

- `chroma_collection_name`: ChromaDB collection name (default: `chroma`)
- `chroma_client_host`: ChromaDB client host
- `chroma_client_port`: ChromaDB client port (default: `8080`)
- `chroma_client_ssl`: Enable/disable SSL (default: `false`)
- `chroma_auth_credentials`: ChromaDB server auth Static API token credentials
- `fields` - Array of fields you want to save. For example, if you want to push `name` and `user.description` fields, you should set this field to `["name", "user.description"]`.
- `metadata_values` - Object of metadata values you want to save. For example, if you want to push `url` and `createdAt` values to ChromaDB, you should set this field to `{"url": "https://www.apify.com", "createdAt": "2021-09-01"}`.
- `metadata_fields` - Object of metadata fields you want to save. For example, if you want to push `url` and `createdAt` fields, you should set this field to `{"url": "url", "createdAt": "createdAt"}`. If it has the same key as `metadata_values`, it's replaced.
- `openai_token` - OpenAI API token.
- `perform_chunking` - Whether to compute text chunks
- `chunk_size` - The maximum character length of each text chunk
- `chunk_overlap` - The character overlap between text chunks that are next to each other

Fields `fields`, `metadata_values`, and `metadata_fields` supports dot notation. For example, if you want to push `name` field from `user` object, you should set `fields` to `["user.name"]`.

## Outputs

This integration will save the selected fields from your Actor to your a ChromaDB.

## Want to talk to other devs or get help?

Join our [developer community on Discord](https://discord.com/invite/jyEM2PRvMU) to connect with other users and discuss this and other integrations.

## Need data for your LLMs?

You can also use the Apify platform to [gather data for your large language models](https://apify.com/data-for-generative-ai). We have Actors to ingest entire websites automatically.
Gather customer documentation, knowledge bases, help centers, forums, blog posts, and other sources of information to train or prompt your LLMs.
Integrate Apify into your product and let your customers upload their content in minutes.
