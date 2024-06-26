"""
Define crawled data for the database playground files
"""

from langchain_core.documents import Document

UUID = "00000000-0000-0000-0000-0000000000"
ID1 = f"{UUID}10"
ID2 = f"{UUID}20"
ID3 = f"{UUID}30"
ID4A, ID4B, ID4C = f"{UUID}4a", f"{UUID}4b", f"{UUID}4c"
ID5A, ID5B, ID5C = f"{UUID}5a", f"{UUID}5b", f"{UUID}5c"
ID6 = f"{UUID}60"

ITEM_ID1 = "id1"
ITEM_ID4 = "id4"

d1 = Document(page_content="Expired->del", metadata={"item_id": ITEM_ID1, "chunk_id": ID1, "checksum": "1", "last_seen_at": 0})
d2 = Document(page_content="Old->not-del", metadata={"item_id": "id2", "chunk_id": ID2, "checksum": "2", "last_seen_at": 1})
d3a = Document(page_content="Unchanged->upt-meta", metadata={"item_id": "id3", "chunk_id": ID3, "checksum": "3", "last_seen_at": 1})
d3b = Document(page_content="Unchanged->upt-meta", metadata={"item_id": "id3", "chunk_id": ID3, "checksum": "3", "last_seen_at": 2})
d4a = Document(page_content="Changed->del", metadata={"item_id": ITEM_ID4, "chunk_id": ID4A, "checksum": "4", "last_seen_at": 1})
d4b = Document(page_content="Changed->del", metadata={"item_id": ITEM_ID4, "chunk_id": ID4B, "checksum": "4", "last_seen_at": 1})
d4c = Document(page_content="Changed->add-new", metadata={"item_id": ITEM_ID4, "chunk_id": ID4C, "checksum": "4c", "last_seen_at": 2})
d5a = Document(page_content="Changed->del", metadata={"item_id": "id5", "chunk_id": ID5A, "checksum": "5", "last_seen_at": 1})
d5b = Document(page_content="Changed->add-new", metadata={"item_id": "id5", "chunk_id": ID5B, "checksum": "5bc", "last_seen_at": 2})
d5c = Document(page_content="Changed->add-new", metadata={"item_id": "id5", "chunk_id": ID5C, "checksum": "5bc", "last_seen_at": 2})
d6 = Document(page_content="New->add", metadata={"item_id": "id5", "chunk_id": ID6, "checksum": "6", "last_seen_at": 2})


crawl_1 = [d1, d2, d3a, d4a, d4b, d5a]
crawl_2 = [d3b, d4c, d5b, d5c, d6]
expected_results = [d2, d3b, d4c, d5b, d5c, d6]
