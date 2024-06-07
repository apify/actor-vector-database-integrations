"""
Define crawled data for the database playground files
"""

from langchain_core.documents import Document

d1 = Document(page_content="Expired->del", metadata={"item_id": "id1", "id": "id1#1", "checksum": "1", "last_seen_at": 0})
d2 = Document(page_content="Old->not-del", metadata={"item_id": "id2", "id": "id2#2", "checksum": "2", "last_seen_at": 1})
d3a = Document(page_content="Unchanged->upt-meta", metadata={"item_id": "id3", "id": "id3#3", "checksum": "3", "last_seen_at": 1})
d3b = Document(page_content="Unchanged->upt-meta", metadata={"item_id": "id3", "id": "id3#3", "checksum": "3", "last_seen_at": 2})
d4a = Document(page_content="Changed->del", metadata={"item_id": "id4", "id": "id4#4a", "checksum": "4", "last_seen_at": 1})
d4b = Document(page_content="Changed->del", metadata={"item_id": "id4", "id": "id4#4b", "checksum": "4", "last_seen_at": 1})
d4c = Document(page_content="Changed->add-new", metadata={"item_id": "id4", "id": "id4#4c", "checksum": "0", "last_seen_at": 2})
d5 = Document(page_content="New->add", metadata={"item_id": "id5", "id": "id5#5", "checksum": "5", "last_seen_at": 2})


crawl_1 = [d1, d2, d3a, d4a, d4b]
crawl_2 = [d3b, d4c, d5]
expected_results = [d2, d3b, d4c, d5]
