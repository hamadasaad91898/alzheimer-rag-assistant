import sys
import json
import subprocess
import traceback

from pathlib import Path


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

PDF_FILE = BASE_DIR / "neurosci.pdf"

DOCLING_JSON_FILE = (
    BASE_DIR / "docling_output.json"
)

DOCLING_MD_FILE = (
    BASE_DIR / "docling_output.md"
)

CLEAN_SCRIPT = (
    BASE_DIR / "clean_docling_json.py"
)

CHUNK_SCRIPT = (
    BASE_DIR / "chunk_docling_json.py"
)

INGEST_SCRIPT = (
    BASE_DIR / "incremental_ingest.py"
)


# =========================================================
# Console helpers
# =========================================================

def print_header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


def print_step(
    number,
    title
):

    print()
    print("-" * 70)

    print(
        f"STEP {number}: {title}"
    )

    print("-" * 70)
    print()


# =========================================================
# Validate project files
# =========================================================

def validate_files():

    required_files = [
        PDF_FILE,
        CLEAN_SCRIPT,
        CHUNK_SCRIPT,
        INGEST_SCRIPT,
    ]

    missing = [
        path.name
        for path in required_files
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing required files: "
            + ", ".join(
                missing
            )
        )


# =========================================================
# Run Python script
# =========================================================

def run_python_script(
    script_path,
    *arguments
):

    command = [
        sys.executable,
        str(
            script_path
        ),
        *arguments,
    ]

    print(
        "Running:",
        " ".join(
            command
        )
    )

    print()

    result = subprocess.run(
        command,
        cwd=str(
            BASE_DIR
        ),
        check=False
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"{script_path.name} failed "
            f"with exit code "
            f"{result.returncode}"
        )


# =========================================================
# Document-level precheck
# =========================================================

def run_document_precheck():

    print_step(
        1,
        "DOCUMENT LEVEL PRECHECK"
    )

    # Import here so the same exact logic
    # from incremental_ingest.py is used.
    from incremental_ingest import (
        run_precheck
    )

    unchanged = run_precheck()

    return bool(
        unchanged
    )


# =========================================================
# Docling conversion
# =========================================================

def run_docling():

    print_step(
        2,
        "DOCLING PDF EXTRACTION"
    )

    try:

        from docling.document_converter import (
            DocumentConverter
        )

    except ImportError as error:

        raise RuntimeError(
            "Docling is not installed.\n"
            "Install it with:\n"
            "pip install docling"
        ) from error


    print(
        "Source:",
        PDF_FILE.name
    )

    print(
        "Starting Docling conversion..."
    )

    print()


    converter = DocumentConverter()


    result = converter.convert(
        PDF_FILE
    )


    document = result.document


    # =====================================================
    # Export JSON
    # =====================================================

    document_dict = (
        document.export_to_dict()
    )


    temporary_json = (
        BASE_DIR
        / "docling_output.tmp.json"
    )


    with open(
        temporary_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            document_dict,
            file,
            ensure_ascii=False,
            indent=2
        )


    # =====================================================
    # Export Markdown
    # =====================================================

    markdown = (
        document.export_to_markdown()
    )


    temporary_md = (
        BASE_DIR
        / "docling_output.tmp.md"
    )


    with open(
        temporary_md,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            markdown
        )


    # =====================================================
    # Replace outputs only after both exports succeed
    # =====================================================

    temporary_json.replace(
        DOCLING_JSON_FILE
    )

    temporary_md.replace(
        DOCLING_MD_FILE
    )


    print(
        "Docling conversion completed"
    )

    print(
        "JSON:",
        DOCLING_JSON_FILE.name
    )

    print(
        "Markdown:",
        DOCLING_MD_FILE.name
    )


# =========================================================
# Cleaning
# =========================================================

def run_cleaning():

    print_step(
        3,
        "CLEANING"
    )

    run_python_script(
        CLEAN_SCRIPT
    )


# =========================================================
# Chunking
# =========================================================

def run_chunking():

    print_step(
        4,
        "SECTION AWARE CHUNKING"
    )

    run_python_script(
        CHUNK_SCRIPT
    )


# =========================================================
# Delta ingestion
# =========================================================

def run_ingestion():

    print_step(
        5,
        "CHUNK LEVEL DELTA INDEXING"
    )

    run_python_script(
        INGEST_SCRIPT
    )


# =========================================================
# Main pipeline
# =========================================================

def main():

    print_header(
        "ALZHEIMER RAG INCREMENTAL INDEX PIPELINE"
    )


    print(
        "Project directory:",
        BASE_DIR
    )

    print(
        "PDF:",
        PDF_FILE.name
    )


    # =====================================================
    # Validate project
    # =====================================================

    validate_files()


    # =====================================================
    # Precheck
    # =====================================================

    unchanged = (
        run_document_precheck()
    )


    # =====================================================
    # PDF unchanged
    # =====================================================

    if unchanged:

        print_header(
            "PIPELINE COMPLETED"
        )

        print(
            "Document status: UNCHANGED"
        )

        print()

        print(
            "Docling:        SKIPPED"
        )

        print(
            "Cleaning:       SKIPPED"
        )

        print(
            "Chunking:       SKIPPED"
        )

        print(
            "Embeddings:     0"
        )

        print(
            "Supabase:       NO CHANGES"
        )

        print()

        print(
            "No processing was required."
        )

        return


    # =====================================================
    # PDF changed
    # =====================================================

    print()
    print(
        "Document changed or index state "
        "requires processing."
    )

    print(
        "Starting full preprocessing pipeline..."
    )


    # =====================================================
    # Docling
    # =====================================================

    run_docling()


    # =====================================================
    # Clean
    # =====================================================

    run_cleaning()


    # =====================================================
    # Chunk
    # =====================================================

    run_chunking()


    # =====================================================
    # Delta ingest
    # =====================================================

    run_ingestion()


    # =====================================================
    # Success
    # =====================================================

    print_header(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )


    print(
        "Document:"
    )

    print(
        f"  {PDF_FILE.name}"
    )


    print()

    print(
        "Completed stages:"
    )

    print(
        "  Document Hash Check"
    )

    print(
        "  Docling Extraction"
    )

    print(
        "  Cleaning"
    )

    print(
        "  Section-Aware Chunking"
    )

    print(
        "  Chunk-Level Delta Indexing"
    )

    print(
        "  Supabase Synchronization"
    )

    print()

    print(
        "Only new or changed chunks "
        "received new embeddings."
    )


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print()
        print(
            "Pipeline cancelled by user."
        )

        sys.exit(
            130
        )

    except Exception as error:

        print()
        print("=" * 70)
        print(
            "PIPELINE FAILED"
        )
        print("=" * 70)

        print()
        print(
            "Error:",
            error
        )

        print()

        traceback.print_exc()

        print()
        print(
            "The successful ingestion state "
            "was not updated."
        )

        sys.exit(
            1
        )