import json
import re
import html


INPUT_FILE = "docling_output.json"
OUTPUT_FILE = "cleaned_docling.json"


# Load Docling JSON
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    document = json.load(f)


source_name = document.get("origin", {}).get(
    "filename",
    "document.pdf"
)


# Create reference map
ref_map = {}

for key in ["texts", "tables", "pictures", "groups"]:
    for item in document.get(key, []):
        ref_map[item["self_ref"]] = item


# Get page numbers
def get_pages(item):
    pages = []

    for prov in item.get("prov", []):
        page_no = prov.get("page_no")

        if page_no is not None:
            pages.append(page_no)

    return sorted(set(pages))


# Clean normal text
def clean_text(text):
    if not text:
        return ""

    text = html.unescape(text)

    text = text.replace("\u00ad", "")
    text = text.replace("\ufffd", "")

    # Remove table continuation text
    text = re.sub(
        r"^Continued on next page\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove reference citations
    # Examples:
    # [71]
    # [90,91]
    # [94-97]
    # [94–97]
    # [7,8,36]
    text = re.sub(
        r"\[\s*\d+(?:\s*[-–—,]\s*\d+)*\s*\]",
        "",
        text
    )

    # Remove spaces left before punctuation
    text = re.sub(
        r"\s+([.,;:])",
        r"\1",
        text
    )

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Check unwanted content
def is_noise(text, label):
    if not text:
        return True

    # Remove headers and footers
    if label in ["page_header", "page_footer"]:
        return True

    # Remove document type heading
    if (
        label == "section_header"
        and text.lower() == "review"
    ):
        return True

    # Remove journal website
    if text.lower() == (
        "https://www.aimspress.com/journal/neuroscience"
    ):
        return True

    # Remove journal line
    if re.fullmatch(
        r"AIMS Neuroscience,\s*13\(2\):\s*208[-–]243\.?",
        text,
        flags=re.IGNORECASE
    ):
        return True

    # Remove article metadata
    if re.match(
        r"^(DOI|Received|Revised|Accepted|Published):",
        text,
        flags=re.IGNORECASE
    ):
        return True

    # Remove image placeholder
    if text == "<!-- image -->":
        return True

    return False


# Convert table to Markdown
def table_to_markdown(table):
    grid = table.get("data", {}).get("grid", [])

    if not grid:
        return ""

    rows = []

    for row in grid:
        values = []

        for cell in row:
            if not cell:
                values.append("")
                continue

            value = clean_text(
                cell.get("text", "")
            )

            value = value.replace("|", r"\|")
            value = value.replace("\n", "<br>")

            values.append(value)

        rows.append(values)

    if not rows:
        return ""

    lines = []

    # Add table header
    lines.append(
        "| " + " | ".join(rows[0]) + " |"
    )

    lines.append(
        "| "
        + " | ".join(["---"] * len(rows[0]))
        + " |"
    )

    # Add table rows
    for row in rows[1:]:
        lines.append(
            "| " + " | ".join(row) + " |"
        )

    return "\n".join(lines)


# Resolve nested items
def resolve_ref(ref, seen):
    if ref in seen:
        return []

    seen.add(ref)

    item = ref_map.get(ref)

    if item is None:
        return []

    # Read group children
    if ref.startswith("#/groups/"):
        result = []

        for child in item.get("children", []):
            child_ref = child.get("$ref")

            if child_ref:
                result.extend(
                    resolve_ref(
                        child_ref,
                        seen
                    )
                )

        return result

    # Keep image captions only
    if ref.startswith("#/pictures/"):
        result = []

        for caption in item.get("captions", []):
            caption_ref = caption.get("$ref")

            if caption_ref:
                result.extend(
                    resolve_ref(
                        caption_ref,
                        seen
                    )
                )

        return result

    return [item]


# Read document in order
ordered_items = []
seen_refs = set()

for child in document.get(
    "body", {}
).get(
    "children", []
):
    ref = child.get("$ref")

    if ref:
        ordered_items.extend(
            resolve_ref(
                ref,
                seen_refs
            )
        )


# Sections not needed for RAG
SKIP_SECTIONS = {
    "use of ai tools declaration",
    "conflict of interest",
    "authors' contributions"
}


clean_items = []

references_found = False
skip_current_section = False


# Clean document items
for item in ordered_items:
    label = item.get("label", "")

    # Handle section headers
    if label == "section_header":
        text = clean_text(
            item.get("text", "")
        )

        # Stop before references
        if text.lower() == "references":
            references_found = True
            break

        # Skip administrative sections
        skip_current_section = (
            text.lower() in SKIP_SECTIONS
        )

        if skip_current_section:
            continue

        if is_noise(text, label):
            continue

        clean_items.append({
            "type": "section_header",
            "text": text,
            "pages": get_pages(item),
            "level": item.get("level", 1)
        })

        continue

    # Skip content inside unwanted sections
    if skip_current_section:
        continue

    # Handle tables
    if label == "table":
        table_text = table_to_markdown(item)

        if table_text:
            clean_items.append({
                "type": "table",
                "text": table_text,
                "pages": get_pages(item)
            })

        continue

    text = clean_text(
        item.get("text", "")
    )

    if is_noise(text, label):
        continue

    # Remove affiliations
    if (
        "University of Milan" in text
        or "Humanitas University" in text
    ):
        continue

    # Remove correspondence
    if text.lower().startswith("correspondence:"):
        continue

    clean_items.append({
        "type": "text",
        "text": text,
        "pages": get_pages(item)
    })


# Save cleaned document
output = {
    "source": source_name,
    "items": clean_items
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )


# Show results
text_count = sum(
    1
    for item in clean_items
    if item["type"] == "text"
)

heading_count = sum(
    1
    for item in clean_items
    if item["type"] == "section_header"
)

table_count = sum(
    1
    for item in clean_items
    if item["type"] == "table"
)


print("=" * 50)
print("Cleaning completed")
print("=" * 50)

print("Source:", source_name)
print("Total items:", len(clean_items))
print("Text items:", text_count)
print("Headings:", heading_count)
print("Tables:", table_count)
print("References removed:", references_found)

print()
print("Saved to:", OUTPUT_FILE)