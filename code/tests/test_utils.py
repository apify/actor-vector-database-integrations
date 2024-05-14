from unittest.mock import patch

from langchain.docstore.document import Document

from utils import get_nested_value, load_dataset, stringify_dict


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

    loader = load_dataset("1234", ["text"], {}, {})
    result = list(map(loader.dataset_mapping_function, dataset_items))

    assert result == [Document(**{"page_content": "text: This is a test"})]


def test_load_page_content_with_metadata():

    dataset_items = [
        {"text": "This is a test", "url": "https://example.com", "metadata": {"title": "Test Title"}},
        {"text": "Another test", "url": "https://example2.com", "metadata": {"title": "Test Title 2"}},
    ]

    meta_values = {"source": "test source"}
    meta_fields = {"page_url": "url", "page_title": "metadata.title"}

    loader = load_dataset("1234", ["text", "url"], meta_values, meta_fields)
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
