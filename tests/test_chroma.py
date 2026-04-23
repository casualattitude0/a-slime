import chromadb
client = chromadb.Client()
collection = client.create_collection("test")
collection.add(ids=["1"], documents=["test"], metadatas=[{"session_id": "test"}])
print(collection.get(where={"session_id": {"$eq": "test"}}))
