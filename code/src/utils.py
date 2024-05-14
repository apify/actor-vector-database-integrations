import copy

from langchain_community.document_loaders import ApifyDatasetLoader
from langchain_core.documents import Document


def get_nested_value(d: dict, keys: str) -> str:
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
    return d  # type ignore


def stringify_dict(d: dict, keys: list[str]) -> str:
    """Stringify all values in a dictionary.

    Example:
        >>> d_ = {"a": {"text": "Apify is cool"}, "description": "Apify platform"}
        >>> stringify_dict(d_, ["a.text", "description"])
        a.text: Apify is cool
        description: Apify platform
    """
    result = []
    for key in keys:
        if value := get_nested_value(d, key):
            result.append(f"{key}: {value}")
    return "\n".join(result)


def load_dataset(dataset_id: str, fields: list[str], meta_values: dict, meta_fields: dict) -> ApifyDatasetLoader:
    """Load dataset by dataset_id using ApifyDatasetLoader.

    The dataset_mapping_function is used to map the dataset item to a Document object.
    Stringify dict using the fields.
    """

    loader_ = ApifyDatasetLoader(
        str(dataset_id),
        dataset_mapping_function=lambda dataset_item: Document(
            page_content=stringify_dict(dataset_item, fields) or "",
            metadata={
                **meta_values,
                **{key: get_nested_value(dataset_item, value) for key, value in meta_fields.items()},
            },
        ),
    )
    return loader_