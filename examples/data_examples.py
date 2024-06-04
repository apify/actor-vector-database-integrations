from typing import Callable

from langchain_core.documents import Document

# Define the Document instances
d1 = Document(page_content="Orphaned->del", metadata={"item_id": "id1", "id": "id1#1", "checksum": "1", "last_seen_at": 0})
d2 = Document(page_content="Old->not-del", metadata={"item_id": "id2", "id": "id2#2", "checksum": "2", "last_seen_at": 1})
d3 = Document(page_content="Unchanged->upt-meta", metadata={"item_id": "id3", "id": "id3#3", "checksum": "3", "last_seen_at": 1})
d4a = Document(page_content="Changed->del", metadata={"item_id": "id4", "id": "id4#4", "checksum": "4", "last_seen_at": 1})
d4b = Document(page_content="Changed->del->add-new", metadata={"item_id": "id4", "id": "id4#6", "checksum": "0", "last_seen_at": 2})
d5 = Document(page_content="New->add", metadata={"item_id": "id5", "id": "id5#5", "checksum": "5", "last_seen_at": 2})

# Define the crawl lists
crawl_1 = [d1, d2, d3, d4a]

# doc1 not crawled -> orphaned -> delete
# doc2 not crawled -> but not orphaned
crawl_2 = [d3, d4b, d5]

# Define the expected results
expected_results = [d2, d3, d4b, d5]


def compare_crawl_with_db(data, search_function: Callable, dummy_search_vector, top_k=10_000):
    """Compare current crawled data with the data in the database. Return data to add, delete and update.

    New data is added
    Data that was not changed -> update metadata updated_at
    Data that was changed -> delete and add new
    """

    _add, _delete, _update_meta = [], [], []
    for d in data:
        res_ = search_function(dummy_search_vector, k=top_k, filter={"item_id": d.metadata["item_id"]})
        if res_ and isinstance(res_[0], tuple):
            res_ = [r_[0] for r_ in res_]
        if res_:
            if d.metadata["checksum"] in {r_.metadata["checksum"] for r_ in res_}:
                _update_meta.append(d)
            else:
                _delete.extend(res_)
                _add.append(d)
        else:
            _add.append(d)

    return _add, _update_meta, _delete