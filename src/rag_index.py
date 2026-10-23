
from pathlib import Path
import re
import time

BASE_DIR = Path(__file__).resolve().parent.parent
GUIDELINES_PATH = BASE_DIR / "data" / "mrna_safety_guidelines.txt"


def load_guidelines():
    """Load the local safety guideline text file."""
    if not GUIDELINES_PATH.exists():
        raise FileNotFoundError(
            f"Guideline file not found: {GUIDELINES_PATH}"
        )

    text = GUIDELINES_PATH.read_text(encoding="utf-8")
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def query_guidelines(question: str, k: int = 3):
    """Retrieve relevant guideline paragraphs using simple keyword matching."""

    start_time = time.perf_counter()
    chunks = load_guidelines()

    if not question or not question.strip():
        return [], time.perf_counter() - start_time

    query_words = set(re.findall(r"\b[a-zA-Z0-9]+\b", question.lower()))

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "what",
        "does", "do", "of", "to", "and", "or", "in", "on", "for",
        "about", "how", "can", "this", "that", "it", "with"
    }
    query_words -= stop_words

    scored_chunks = []

    for chunk in chunks:
        chunk_words = set(
            re.findall(r"\b[a-zA-Z0-9]+\b", chunk.lower())
        )
        score = len(query_words & chunk_words)

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    results = [chunk for _, chunk in scored_chunks[:k]]

    elapsed = time.perf_counter() - start_time
    return results, elapsed
