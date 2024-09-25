from __future__ import annotations

import copy

from langchain_core.documents import Document

from src.utils import (
    add_item_checksum,
    compute_hash,
    get_chunks_to_delete,
    get_chunks_to_update,
    get_dataset_loader,
    get_nested_value,
    stringify_dict,
)


def test_get_nested_value_with_nested_keys() -> None:
    d = {"a": {"b": {"c": "value"}}}
    assert get_nested_value(d, "a.b.c") == "value"


def test_get_nested_value_with_top_level_key() -> None:
    d = {"a": "value"}
    assert get_nested_value(d, "a") == "value"


def test_get_nested_value_with_nonexistent_key() -> None:
    d = {"a": "value"}
    assert get_nested_value(d, "b") == ""


def test_get_nested_value_with_empty_dict() -> None:
    assert get_nested_value({}, "a") == ""


def test_stringify_dict_with_multiple_keys() -> None:
    d = {"a": "value1", "b": "value2"}
    keys = ["a", "b"]
    assert stringify_dict(d, keys) == "a: value1\nb: value2"


def test_stringify_dict_with_nested_keys() -> None:
    d = {"a": {"b": "value"}}
    keys = ["a.b"]
    assert stringify_dict(d, keys) == "a.b: value"


def test_stringify_dict_with_nonexistent_keys() -> None:
    d = {"a": "value"}
    keys = ["b"]
    assert stringify_dict(d, keys) == ""


def test_stringify_dict_with_empty_dict() -> None:
    assert stringify_dict({}, ["a"]) == ""


def test_load_page_content() -> None:
    dataset_items = [{"text": "This is a test"}]

    loader = get_dataset_loader("1234", ["text"], {}, {})
    result = list(map(loader.dataset_mapping_function, dataset_items))

    assert result == [Document(page_content="text: This is a test")]


def test_load_page_content_with_metadata() -> None:
    dataset_items = [
        {"text": "This is a test", "url": "https://example.com", "metadata": {"title": "Test Title"}},
        {"text": "Another test", "url": "https://example2.com", "metadata": {"title": "Test Title 2"}},
    ]

    meta_values = {"source": "test source"}
    meta_fields = {"page_url": "url", "page_title": "metadata.title"}

    loader = get_dataset_loader("1234", ["text", "url"], meta_values, meta_fields)
    result = list(map(loader.dataset_mapping_function, dataset_items))

    expected_result = [
        Document(
            page_content="text: This is a test\nurl: https://example.com",
            metadata={"source": "test source", "page_url": "https://example.com", "page_title": "Test Title"},
        ),
        Document(
            page_content="text: Another test\nurl: https://example2.com",
            metadata={"source": "test source", "page_url": "https://example2.com", "page_title": "Test Title 2"},
        ),
    ]

    assert result == expected_result


def test_compute_hash() -> None:
    text = "test"
    assert compute_hash(text) == "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"


def test_get_chunks_empty() -> None:
    add_, update_ = get_chunks_to_update([], [])
    assert len(add_) == 0
    assert len(update_) == 0


def test_get_chunks_previous_run_empty(documents: list[Document]) -> None:
    add_, update_ = get_chunks_to_update([], documents)
    assert len(add_) == 1
    assert len(update_) == 0

    assert add_[0].metadata["item_id"] == documents[0].metadata["item_id"]


def test_get_chunks_current_run_empty(documents: list[Document]) -> None:
    add_, update_ = get_chunks_to_update(documents, [])
    assert len(add_) == 0
    assert len(update_) == 0


def test_get_chunks_update_metadata(documents: list[Document]) -> None:
    chunks = add_item_checksum(documents, ["url"])

    add_, update_ = get_chunks_to_update(chunks, chunks)
    assert len(add_) == 0
    assert len(update_) == 1
    assert update_[0].metadata["checksum"] == "0feb3e25afe9430e2d23d726cbe2cecccef2afff29cdecf7a747264433605fa4"
    assert update_[0].metadata["item_id"] == "f2881510b05f8c3567c1d63a3212d3ebb8bbfc5510241db1f39da8f66df1defd"


def test_get_chunks_to_update_with_content_changes(documents: list[Document]) -> None:
    chunks_prev = add_item_checksum(documents, ["url"])

    chunks_curr = copy.deepcopy(chunks_prev)
    chunks_curr[0].page_content = "Content has changed between runs"
    chunks_curr = add_item_checksum(chunks_curr, ["url"])

    assert chunks_prev[0].metadata["item_id"] == chunks_curr[0].metadata["item_id"]
    assert chunks_prev[0].metadata["checksum"] != chunks_curr[0].metadata["checksum"]

    add_, update_ = get_chunks_to_update(chunks_prev, chunks_curr)
    assert len(add_) == 1
    assert len(update_) == 0
    assert add_[0] == chunks_curr[0]


def test_get_chunks_to_delete_empty() -> None:
    chunks_prev = add_item_checksum([], ["url"])
    delete_, old_keep_ = get_chunks_to_delete(chunks_prev, chunks_prev, 1)
    assert len(delete_) == 0
    assert len(old_keep_) == 0


def test_get_chunks_to_delete_no_delete(documents: list[Document]) -> None:
    chunks_prev = add_item_checksum(documents, ["url"])
    delete_, old_keep_ = get_chunks_to_delete(chunks_prev, chunks_prev, 1)
    assert len(delete_) == 0
    assert len(old_keep_) == 0


def test_get_chunks_to_delete_delete_expired(documents: list[Document]) -> None:
    chunks_prev = add_item_checksum(documents, ["url"])
    chunks_prev[0].metadata["last_seen_at"] = 1

    delete_, old_keep = get_chunks_to_delete(chunks_prev, [], 1)
    assert len(delete_) == 1
    assert len(old_keep) == 0
    assert delete_[0] == chunks_prev[0]
