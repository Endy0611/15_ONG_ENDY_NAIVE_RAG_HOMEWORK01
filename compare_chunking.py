"""
Bonus: compare two chunking strategies on the same documents.

Run: poetry run python compare_chunking.py
"""

from app.config import DATA_DIR
from app.ingestion import chunk_text, chunk_text_by_paragraph, load_documents


def _summarize(chunks: list[str]) -> tuple[int, float]:
    if not chunks:
        return 0, 0.0
    lengths = [len(c) for c in chunks]
    return len(chunks), sum(lengths) / len(lengths)


def main() -> None:
    documents = load_documents(DATA_DIR)
    if not documents:
        print(f"No documents found in '{DATA_DIR}/'.")
        return

    for filename, text in documents:
        fixed = chunk_text(text)
        paragraph = chunk_text_by_paragraph(text)
        fixed_count, fixed_avg = _summarize(fixed)
        para_count, para_avg = _summarize(paragraph)

        print(f"=== {filename} ===")
        print(f"  Fixed-size:       {fixed_count} chunks, avg {fixed_avg:.0f} chars")
        print(f"  Paragraph-aware:  {para_count} chunks, avg {para_avg:.0f} chars")
        print(f"  Fixed-size[0]:      {fixed[0][:100]!r}...")
        print(f"  Paragraph-aware[0]: {paragraph[0][:100]!r}...")
        print()


if __name__ == "__main__":
    main()