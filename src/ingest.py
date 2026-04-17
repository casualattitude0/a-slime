import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_google_key() -> str | None:
    load_dotenv(_project_root() / ".env")
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


def _embedding_model_name() -> str:
    return os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")


def ingest(data_dir: Path, chroma_dir: Path, chunk_size: int, chunk_overlap: int) -> int:
    api_key = _require_google_key()
    if not api_key:
        env_path = _project_root() / ".env"
        print(
            "Missing API key: set GOOGLE_API_KEY or GEMINI_API_KEY in "
            f"{env_path} (see .env.example)",
            file=sys.stderr,
        )
        return 1

    loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs = loader.load()
    if not docs:
        print(f"No PDF files found under {data_dir}", file=sys.stderr)
        return 1

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)

    embed_model = _embedding_model_name()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embed_model,
        google_api_key=api_key,
    )
    chroma_dir.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(chroma_dir),
    )
    print(
        f"Ingested {len(docs)} document(s) into {len(splits)} chunk(s) at {chroma_dir} "
        f"(embedding model: {embed_model})"
    )
    return 0


def main() -> None:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Embed PDFs into ChromaDB.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data",
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=root / "chroma_db",
        help="Chroma persistence directory",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    sys.exit(
        ingest(
            data_dir=args.data_dir.resolve(),
            chroma_dir=args.chroma_dir.resolve(),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    )


if __name__ == "__main__":
    main()
