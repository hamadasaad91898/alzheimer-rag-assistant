

import json
from docling.document_converter import DocumentConverter

# Create the converter
converter = DocumentConverter()

# Convert the PDF
result = converter.convert("neurosci.pdf")

# Save as Markdown
markdown_text = result.document.export_to_markdown()

with open("docling_output.md", "w", encoding="utf-8") as f:
    f.write(markdown_text)

# Save as JSON to keep document metadata
document_data = result.document.export_to_dict()

with open("docling_output.json", "w", encoding="utf-8") as f:
    json.dump(
        document_data,
        f,
        ensure_ascii=False,
        indent=2
    )

print("Conversion completed")
print("Markdown saved: docling_output.md")
print("JSON saved: docling_output.json")