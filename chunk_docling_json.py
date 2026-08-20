import json
import tiktoken


INPUT_FILE = "cleaned_docling.json"
OUTPUT_FILE = "chunks.json"

MAX_TOKENS = 700
OVERLAP_TOKENS = 100

encoding = tiktoken.get_encoding("cl100k_base")


# Load cleaned JSON
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    document = json.load(f)


source_name = document.get(
    "source",
    "document.pdf"
)

items = document.get("items", [])


# Count tokens
def count_tokens(text):
    return len(
        encoding.encode(text)
    )


# Create a chunk
def create_chunk(section, chunk_type, text, pages):
    content = (
        f"Section: {section}\n\n{text}"
    )

    return {
        "source": source_name,
        "section": section,
        "pages": sorted(set(pages)),
        "type": chunk_type,
        "content": content,
        "token_count": count_tokens(content)
    }


chunks = []

current_section = "Document"
current_text = ""
current_pages = []


# Save current text chunk
def save_current_chunk():
    global current_text
    global current_pages

    if not current_text.strip():
        return

    chunks.append(
        create_chunk(
            current_section,
            "text",
            current_text.strip(),
            current_pages
        )
    )

    current_text = ""
    current_pages = []


# Process document items
for item in items:
    item_type = item.get("type", "")
    text = item.get("text", "").strip()
    pages = item.get("pages", [])

    if not text:
        continue

    # Update current section
    if item_type == "section_header":
        save_current_chunk()

        current_section = text
        continue

    # Keep tables as separate chunks
    if item_type == "table":
        save_current_chunk()

        chunks.append(
            create_chunk(
                current_section,
                "table",
                text,
                pages
            )
        )

        continue

    # Count current text item
    text_tokens = encoding.encode(text)

    # Split a very large text block
    if len(text_tokens) > MAX_TOKENS:
        save_current_chunk()

        start = 0

        while start < len(text_tokens):
            end = min(
                start + MAX_TOKENS,
                len(text_tokens)
            )

            piece_tokens = text_tokens[start:end]

            piece_text = encoding.decode(
                piece_tokens
            ).strip()

            if piece_text:
                chunks.append(
                    create_chunk(
                        current_section,
                        "text",
                        piece_text,
                        pages
                    )
                )

            if end >= len(text_tokens):
                break

            start = end - OVERLAP_TOKENS

        continue

    # Start a new chunk
    if not current_text:
        current_text = text
        current_pages = list(pages)

        continue

    # Try to add the next paragraph
    candidate = (
        current_text
        + "\n\n"
        + text
    )

    if count_tokens(candidate) <= MAX_TOKENS:
        current_text = candidate

        current_pages = sorted(
            set(
                current_pages
                + pages
            )
        )

    else:
        save_current_chunk()

        current_text = text
        current_pages = list(pages)


# Save last chunk
save_current_chunk()


# Add chunk IDs
final_chunks = []

for index, chunk in enumerate(
    chunks,
    start=1
):
    final_chunks.append({
        "chunk_id": index,
        "source": chunk["source"],
        "section": chunk["section"],
        "pages": chunk["pages"],
        "type": chunk["type"],
        "content": chunk["content"],
        "token_count": chunk["token_count"]
    })


# Save chunks
with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        final_chunks,
        f,
        ensure_ascii=False,
        indent=2
    )


# Show statistics
text_chunks = sum(
    1
    for chunk in final_chunks
    if chunk["type"] == "text"
)

table_chunks = sum(
    1
    for chunk in final_chunks
    if chunk["type"] == "table"
)

token_counts = [
    chunk["token_count"]
    for chunk in final_chunks
]


print("=" * 50)
print("Chunking completed")
print("=" * 50)

print("Total chunks:", len(final_chunks))
print("Text chunks:", text_chunks)
print("Table chunks:", table_chunks)

if token_counts:
    print(
        "Average tokens:",
        round(
            sum(token_counts)
            / len(token_counts)
        )
    )

    print(
        "Smallest chunk:",
        min(token_counts)
    )

    print(
        "Largest chunk:",
        max(token_counts)
    )

print()
print("Saved to:", OUTPUT_FILE)