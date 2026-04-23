from langchain_community.vectorstores import Chroma
store = Chroma(persist_directory="chroma_db", collection_name="chat_history")
print(store._collection.get(where={"session_id": {"$eq": "ebf87559-662e-4f13-a4fa-36550e97d778"}}))
