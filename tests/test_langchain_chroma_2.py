from langchain_community.vectorstores import Chroma
store = Chroma(persist_directory="chroma_db", collection_name="chat_history")
print(store._collection.get(include=["metadatas"]))
