import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import build_executor, normalize_agent_output


def main() -> None:
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print("Set GOOGLE_API_KEY or GEMINI_API_KEY in .env (required for RAG embeddings).", file=sys.stderr)
        sys.exit(1)
    try:
        agent = build_executor()
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print("Ctrl+C or empty EOF to exit.")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        result = agent.invoke({"input": user_input, "chat_history": []})
        out = result.get("output")
        text = normalize_agent_output(out)
        if text:
            print(text)
        else:
            print("No text output; full result:", result, file=sys.stderr)


if __name__ == "__main__":
    main()
