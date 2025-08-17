from __future__ import annotations

import copy
import hashlib
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_apify import ApifyDatasetLoader
from langchain_core.documents import Document

EXCLUDE_KEYS_FROM_CHECKSUM = {"metadata": {"chunk_id", "id", "checksum", "last_seen_at", "item_id"}}
DAY_IN_SECONDS = 24 * 3600

logger = logging.getLogger("apify")


def get_nested_value(d: dict, keys: str) -> Any:
    """
    Extract nested value from dict.

    Example:
      >>> get_nested_value({"a": "v1", "c1": {"c2": "v2"}}, "c1.c2")
      'v2'
    """

    d = copy.deepcopy(d)
    for key in keys.split("."):
        if d and isinstance(d, dict) and d.get(key):
            d = d[key]
        else:
            return ""
    return d


def stringify_dict(d: dict, keys: list[str]) -> str:
    """Stringify all values in a dictionary.

    Example:
        >>> d_ = {"a": {"text": "Apify is cool"}, "description": "Apify platform"}
        >>> stringify_dict(d_, ["a.text", "description"])
        'a.text: Apify is cool\\ndescription: Apify platform'
    """
    return "\n".join([f"{key}: {value}" for key in keys if (value := get_nested_value(d, key))])


def get_dataset_loader(dataset_id: str, fields: list[str], meta_object: dict, meta_fields: dict) -> ApifyDatasetLoader:
    """Load dataset by dataset_id using ApifyDatasetLoader.

    The dataset_mapping_function is used to map the dataset item to a Document object.
    Stringify dict using the fields.
    """

    return ApifyDatasetLoader(
        dataset_id,
        dataset_mapping_function=lambda dataset_item: Document(
            page_content=stringify_dict(dataset_item, fields) or "",
            metadata={
                **meta_object,
                **{key: get_nested_value(dataset_item, value) for key, value in meta_fields.items()},
            },
        ),
    )


def compute_hash(text: str) -> str:
    """Compute hash of the text."""
    return hashlib.sha256(text.encode()).hexdigest()


def get_chunks_to_delete(chunks_prev: list[Document], chunks_current: list[Document], expired_days: float) -> tuple[list[Document], list[Document]]:
    """
    Identifies chunks to be deleted based on their last seen timestamp and presence in the current run.

    Compare the chunks from the previous and current runs and identify chunks that are not present
    in the current run and have not been updated within the specified 'expired_days'. These chunks are marked for deletion.
    """
    ids_current = {d.metadata["item_id"] for d in chunks_current}

    ts_expired = int(datetime.now(timezone.utc).timestamp() - expired_days * DAY_IN_SECONDS)
    chunks_expired_delete, chunks_old_keep = [], []

    # chunks that have been crawled in the current run and are older than ts_expired => to delete
    for d in chunks_prev:
        if d.metadata["item_id"] not in ids_current:
            if d.metadata["last_seen_at"] < ts_expired:
                chunks_expired_delete.append(d)
            else:
                chunks_old_keep.append(d)

    return chunks_expired_delete, chunks_old_keep


def get_chunks_to_update(chunks_prev: list[Document], chunks_current: list[Document]) -> tuple[list[Document], list[Document]]:
    """
    Identifies chunks that need to be updated or added based on their unique identifiers and checksums.

    Compare the chunks from the previous and current runs and identify chunks that are new or have
    undergone content changes by comparing their checksums. These chunks are marked for addition. chunks that are
    present in both runs but have not undergone content changes are marked for metadata update.
    """

    prev_id_checksum = defaultdict(list)
    for chunk in chunks_prev:
        prev_id_checksum[chunk.metadata["item_id"]].append(chunk.metadata["checksum"])

    chunks_add = []
    chunks_update_metadata = []
    for chunk in chunks_current:
        if chunk.metadata["item_id"] in prev_id_checksum:
            if chunk.metadata["checksum"] in prev_id_checksum[chunk.metadata["item_id"]]:
                chunks_update_metadata.append(chunk)
            else:
                chunks_add.append(chunk)
        else:
            chunks_add.append(chunk)

    return chunks_add, chunks_update_metadata


def add_item_last_seen_at(items: list[Document]) -> list[Document]:
    """Add last_seen_at timestamp to the metadata of each dataset item."""
    for item in items:
        item.metadata["last_seen_at"] = int(datetime.now(timezone.utc).timestamp())
    return items


def add_item_checksum(items: list[Document], dataset_fields_to_item_id: list[str]) -> list[Document]:
    """
    Adds a checksum and unique item_id to the metadata of each dataset item.

    This function computes a checksum for each item based on its content and metadata, excluding certain keys.
    The checksum is then added to the document's metadata. Additionally, a unique item ID is generated based on
    specified keys in the document's metadata and added to the metadata as well.
    """
    for item in items:
        item.metadata["checksum"] = compute_hash(item.json(exclude=EXCLUDE_KEYS_FROM_CHECKSUM))
        hash_str = "".join([str(item.metadata[key]) for key in dataset_fields_to_item_id])
        item.metadata["item_id"] = compute_hash(hash_str)
        if not hash_str:
            logger.warning(
                "Item_id %s was generated with an empty hash. This typically means that `dataUpdatesPrimaryDatasetFields` "
                "are empty or non-existent.",
                item.metadata["item_id"],
            )

    return add_item_last_seen_at(items)


def add_chunk_id(chunks: list[Document]) -> list[Document]:
    """For every chunk (document stored in vector db) add chunk_id to metadata.

    The chunk_id is a unique identifier for each chunk and is not required, but it is better to keep it in metadata.
    """
    for d in chunks:
        d.metadata["chunk_id"] = d.metadata.get("chunk_id", str(uuid4()))
    return chunks
