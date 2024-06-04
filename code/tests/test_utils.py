import copy

from langchain_core.documents import Document

from store_vector_db.utils import (
    add_item_checksum,
    compute_hash,
    get_chunks_to_delete,
    get_chunks_to_update,
    get_dataset_loader,
    get_nested_value,
    stringify_dict,
)


def test_get_nested_value_with_nested_keys():
    d = {"a": {"b": {"c": "value"}}}
    assert get_nested_value(d, "a.b.c") == "value"


def test_get_nested_value_with_top_level_key():
    d = {"a": "value"}
    assert get_nested_value(d, "a") == "value"


def test_get_nested_value_with_nonexistent_key():
    d = {"a": "value"}
    assert get_nested_value(d, "b") == ""


def test_get_nested_value_with_empty_dict():
    d = {}
    assert get_nested_value(d, "a") == ""


def test_stringify_dict_with_multiple_keys():
    d = {"a": "value1", "b": "value2"}
    keys = ["a", "b"]
    assert stringify_dict(d, keys) == "a: value1\nb: value2"


def test_stringify_dict_with_nested_keys():
    d = {"a": {"b": "value"}}
    keys = ["a.b"]
    assert stringify_dict(d, keys) == "a.b: value"


def test_stringify_dict_with_nonexistent_keys():
    d = {"a": "value"}
    keys = ["b"]
    assert stringify_dict(d, keys) == ""


def test_stringify_dict_with_empty_dict():
    d = {}
    keys = ["a"]
    assert stringify_dict(d, keys) == ""


def test_load_page_content():

    dataset_items = [{"text": "This is a test"}]

    loader = get_dataset_loader("1234", ["text"], {}, {})
    result = list(map(loader.dataset_mapping_function, dataset_items))

    assert result == [Document(**{"page_content": "text: This is a test"})]


def test_load_page_content_with_metadata():

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


def test_compute_hash():
    text = "test"
    assert compute_hash(text) == "098f6bcd4621d373cade4e832627b4f6"


def test_get_chunks_empty():

    add_, update_ = get_chunks_to_update([], [])
    assert len(add_) == 0
    assert len(update_) == 0


def test_get_chunks_previous_run_empty(documents):

    add_, update_ = get_chunks_to_update([], documents)
    assert len(add_) == 1
    assert len(update_) == 0

    assert add_[0].metadata["item_id"] == documents[0].metadata["item_id"]


def test_get_chunks_current_run_empty(documents):

    add_, update_ = get_chunks_to_update(documents, [])
    assert len(add_) == 0
    assert len(update_) == 0


def test_get_chunks_update_metadata(documents):

    chunks = add_item_checksum(documents, ["url"])

    add_, update_ = get_chunks_to_update(chunks, chunks)
    assert len(add_) == 0
    assert len(update_) == 1
    assert update_[0].metadata["checksum"] == "98d743a515d313f1c951710cd9af623c"
    assert update_[0].metadata["item_id"] == "01e53427148c2442b45dc7344b7ed700"


def test_get_chunks_to_update_with_content_changes(documents):

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


def test_get_chunks_to_delete_empty():

    chunks_prev = add_item_checksum([], ["url"])
    delete_, old_keep_ = get_chunks_to_delete(chunks_prev, chunks_prev, 1)
    assert len(delete_) == 0
    assert len(old_keep_) == 0


def test_get_chunks_to_delete_no_delete(documents):

    chunks_prev = add_item_checksum(documents, ["url"])
    delete_, old_keep_ = get_chunks_to_delete(chunks_prev, chunks_prev, 1)
    assert len(delete_) == 0
    assert len(old_keep_) == 0


def test_get_chunks_to_delete_delete_orphaned(documents):

    chunks_prev = add_item_checksum(documents, ["url"])
    chunks_prev[0].metadata["updated_at"] = 1

    delete_, old_keep = get_chunks_to_delete(chunks_prev, [], 1)
    assert len(delete_) == 1
    assert len(old_keep) == 0
    assert delete_[0] == chunks_prev[0]
