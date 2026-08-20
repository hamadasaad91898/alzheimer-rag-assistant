import os
import re
import sys
import json
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client


# =========================================================
# Paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

PRODUCTION_PDF = (
    PROJECT_DIR / "neurosci.pdf"
)

PRODUCTION_SOURCE = "neurosci.pdf"

TEST_SOURCE = (
    "neurosci_incremental_test.pdf"
)

TEST_DIR = (
    PROJECT_DIR
    / "_incremental_test_workspace"
)


FILES_TO_COPY = [
    "index_pipeline.py",
    "incremental_ingest.py",
    "clean_docling_json.py",
    "chunk_docling_json.py",
]


# =========================================================
# Environment
# =========================================================

load_dotenv(
    PROJECT_DIR / ".env"
)


SUPABASE_URL = os.getenv(
    "SUPABASE_URL"
)

SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY"
)


AZURE_OPENAI_API_KEY = os.getenv(
    "AZURE_OPENAI_API_KEY"
)

AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT"
)

AZURE_OPENAI_EMBEDDING_DEPLOYMENT = (
    os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    )
)


required_env = {

    "SUPABASE_URL":
        SUPABASE_URL,

    "SUPABASE_KEY":
        SUPABASE_KEY,

    "AZURE_OPENAI_API_KEY":
        AZURE_OPENAI_API_KEY,

    "AZURE_OPENAI_ENDPOINT":
        AZURE_OPENAI_ENDPOINT,

    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT":
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
}


missing = [

    name

    for name, value
    in required_env.items()

    if not value
]


if missing:

    raise ValueError(
        "Missing environment variables: "
        + ", ".join(
            missing
        )
    )


supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# =========================================================
# Console
# =========================================================

def header(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


def phase(title):

    print()
    print("-" * 80)
    print(title)
    print("-" * 80)
    print()


# =========================================================
# Supabase helpers
# =========================================================

def get_source_rows(
    source
):

    response = (

        supabase
        .table(
            "documents"
        )
        .select(
            "id,"
            "chunk_id,"
            "chunk_hash,"
            "document_hash,"
            "source"
        )
        .eq(
            "source",
            source
        )
        .execute()
    )


    return (
        response.data
        or []
    )


def production_snapshot():

    rows = get_source_rows(
        PRODUCTION_SOURCE
    )


    normalized = sorted(
        [
            {
                "id":
                    row.get(
                        "id"
                    ),

                "chunk_id":
                    row.get(
                        "chunk_id"
                    ),

                "chunk_hash":
                    row.get(
                        "chunk_hash"
                    ),

                "document_hash":
                    row.get(
                        "document_hash"
                    ),

                "source":
                    row.get(
                        "source"
                    ),
            }

            for row
            in rows
        ],

        key=lambda item: (
            str(
                item.get(
                    "id"
                )
            )
        )
    )


    return normalized


def delete_test_rows():

    rows = get_source_rows(
        TEST_SOURCE
    )


    if not rows:

        return 0


    (
        supabase
        .table(
            "documents"
        )
        .delete()
        .eq(
            "source",
            TEST_SOURCE
        )
        .execute()
    )


    return len(
        rows
    )


# =========================================================
# Files
# =========================================================

def validate_project():

    required = [
        PRODUCTION_PDF
    ]


    for filename in FILES_TO_COPY:

        required.append(
            PROJECT_DIR / filename
        )


    missing_files = [

        path.name

        for path
        in required

        if not path.exists()
    ]


    if missing_files:

        raise FileNotFoundError(
            "Missing project files: "
            + ", ".join(
                missing_files
            )
        )


def prepare_workspace():

    if TEST_DIR.exists():

        shutil.rmtree(
            TEST_DIR
        )


    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # =====================================================
    # Copy test PDF
    # =====================================================

    test_pdf = (
        TEST_DIR
        / TEST_SOURCE
    )


    shutil.copy2(
        PRODUCTION_PDF,
        test_pdf
    )


    # =====================================================
    # Copy scripts
    # =====================================================

    for filename in FILES_TO_COPY:

        source_path = (
            PROJECT_DIR
            / filename
        )

        destination_path = (
            TEST_DIR
            / filename
        )


        text = source_path.read_text(
            encoding="utf-8"
        )


        # -------------------------------------------------
        # Test copy must use a different source name.
        # Production scripts remain untouched.
        # -------------------------------------------------

        if filename in {
            "index_pipeline.py",
            "incremental_ingest.py",
        }:

            text = text.replace(
                '"neurosci.pdf"',
                f'"{TEST_SOURCE}"'
            )


        destination_path.write_text(
            text,
            encoding="utf-8"
        )


    return test_pdf


# =========================================================
# Run command
# =========================================================

def run_command(
    command,
    title
):

    phase(
        title
    )


    environment = (
        os.environ.copy()
    )


    result = subprocess.run(
        command,
        cwd=str(
            TEST_DIR
        ),
        env=environment,
        text=True,
        capture_output=True
    )


    if result.stdout:

        print(
            result.stdout
        )


    if result.stderr:

        print(
            result.stderr
        )


    if result.returncode != 0:

        raise RuntimeError(
            f"{title} failed "
            f"with exit code "
            f"{result.returncode}"
        )


    return (
        result.stdout
        or ""
    )


# =========================================================
# Parse console metrics
# =========================================================

def extract_int(
    text,
    label
):

    match = re.search(
        rf"{re.escape(label)}\s*:\s*(\d+)",
        text,
        flags=re.IGNORECASE
    )


    if not match:

        return None


    return int(
        match.group(
            1
        )
    )


# =========================================================
# Make PDF hash change safely
# =========================================================

def change_test_pdf_metadata(
    test_pdf
):

    phase(
        "PHASE C PREPARATION - CHANGE TEST PDF HASH"
    )


    changed = False


    # =====================================================
    # Preferred method
    # Rewrite PDF with a test metadata field.
    # =====================================================

    try:

        from pypdf import (
            PdfReader,
            PdfWriter
        )


        reader = PdfReader(
            str(
                test_pdf
            )
        )


        writer = PdfWriter()


        for page in reader.pages:

            writer.add_page(
                page
            )


        metadata = {}


        if reader.metadata:

            for key, value in (
                reader.metadata.items()
            ):

                if (
                    key
                    and value is not None
                ):

                    metadata[
                        str(
                            key
                        )
                    ] = str(
                        value
                    )


        metadata[
            "/IncrementalIndexTest"
        ] = (
            "document-hash-change"
        )


        writer.add_metadata(
            metadata
        )


        temporary_pdf = (
            TEST_DIR
            / "changed.tmp.pdf"
        )


        with open(
            temporary_pdf,
            "wb"
        ) as file:

            writer.write(
                file
            )


        temporary_pdf.replace(
            test_pdf
        )


        changed = True


        print(
            "PDF metadata changed "
            "using pypdf"
        )


    except ImportError:

        print(
            "pypdf is not installed"
        )


    # =====================================================
    # Fallback
    # PDF permits trailing comments after EOF.
    # This changes the file hash without touching pages.
    # =====================================================

    if not changed:

        with open(
            test_pdf,
            "ab"
        ) as file:

            file.write(
                b"\n% incremental-index-test-marker\n"
            )


        print(
            "PDF hash changed using "
            "a trailing PDF comment"
        )


    print(
        "Test PDF changed safely"
    )


# =========================================================
# Modify exactly one test chunk
# =========================================================

def mutate_one_test_chunk():

    phase(
        "PHASE D PREPARATION - MODIFY ONE TEST CHUNK"
    )


    chunks_file = (
        TEST_DIR
        / "chunks.json"
    )


    if not chunks_file.exists():

        raise FileNotFoundError(
            "Test chunks.json was not found"
        )


    with open(
        chunks_file,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(
            file
        )


    if not chunks:

        raise ValueError(
            "Test chunks.json is empty"
        )


    original_content = (
        chunks[0][
            "content"
        ]
    )


    chunks[0][
        "content"
    ] = (
        original_content
        + "\n\n"
        + "Incremental indexing isolated test marker."
    )


    with open(
        chunks_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


    print(
        "Modified test Chunk ID:",
        chunks[0][
            "chunk_id"
        ]
    )


    print(
        "Production chunks.json was NOT changed"
    )


# =========================================================
# Assertions
# =========================================================

def require(
    condition,
    message
):

    if not condition:

        raise AssertionError(
            message
        )


# =========================================================
# Main test
# =========================================================

def main():

    header(
        "ISOLATED INCREMENTAL INDEXING TEST"
    )


    validate_project()


    production_before = (
        production_snapshot()
    )


    print(
        "Production rows before test:",
        len(
            production_before
        )
    )


    require(
        len(
            production_before
        ) > 0,
        (
            "Production neurosci.pdf rows "
            "were not found in Supabase"
        )
    )


    # =====================================================
    # Remove stale test data only
    # =====================================================

    stale_deleted = (
        delete_test_rows()
    )


    if stale_deleted:

        print(
            "Removed stale test rows:",
            stale_deleted
        )


    test_pdf = (
        prepare_workspace()
    )


    try:

        # =================================================
        # PHASE A
        #
        # Brand new test document.
        # Full pipeline should run.
        # =================================================

        output_a = run_command(
            [
                sys.executable,
                str(
                    TEST_DIR
                    / "index_pipeline.py"
                )
            ],
            (
                "PHASE A - INITIAL TEST DOCUMENT INDEX"
            )
        )


        test_rows_a = (
            get_source_rows(
                TEST_SOURCE
            )
        )


        require(
            len(
                test_rows_a
            ) > 0,
            (
                "Phase A failed: "
                "no test rows were created"
            )
        )


        print(
            "Test rows after Phase A:",
            len(
                test_rows_a
            )
        )


        # =================================================
        # PHASE B
        #
        # Run exactly the same PDF again.
        # Entire preprocessing should be skipped.
        # =================================================

        output_b = run_command(
            [
                sys.executable,
                str(
                    TEST_DIR
                    / "index_pipeline.py"
                )
            ],
            (
                "PHASE B - UNCHANGED DOCUMENT"
            )
        )


        require(
            "Document status: UNCHANGED"
            in output_b,
            (
                "Phase B failed: "
                "document was not recognized "
                "as unchanged"
            )
        )


        require(
            "Docling:        SKIPPED"
            in output_b,
            (
                "Phase B failed: "
                "Docling was not skipped"
            )
        )


        require(
            "Embeddings:     0"
            in output_b,
            (
                "Phase B failed: "
                "expected zero embeddings"
            )
        )


        # =================================================
        # PHASE C
        #
        # Change only the PDF file hash.
        # Upstream preprocessing must run again.
        # Extracted text should remain the same.
        # Therefore chunk embeddings should be reused.
        # =================================================

        change_test_pdf_metadata(
            test_pdf
        )


        output_c = run_command(
            [
                sys.executable,
                str(
                    TEST_DIR
                    / "index_pipeline.py"
                )
            ],
            (
                "PHASE C - CHANGED DOCUMENT HASH"
            )
        )


        require(
            "Starting full preprocessing pipeline"
            in output_c,
            (
                "Phase C failed: "
                "changed PDF did not trigger "
                "processing"
            )
        )


        require(
            "Docling conversion completed"
            in output_c,
            (
                "Phase C failed: "
                "Docling did not run"
            )
        )


        embeddings_c = extract_int(
            output_c,
            "Embeddings created"
        )


        print(
            "Phase C embeddings created:",
            embeddings_c
        )


        # Metadata-only PDF change should normally
        # keep extracted chunks identical.
        require(
            embeddings_c == 0,
            (
                "Phase C expected 0 embeddings "
                "because page content was unchanged"
            )
        )


        # =================================================
        # PHASE D
        #
        # Change exactly one temporary chunk.
        # Only one embedding should be generated.
        # =================================================

        mutate_one_test_chunk()


        output_d = run_command(
            [
                sys.executable,
                str(
                    TEST_DIR
                    / "incremental_ingest.py"
                )
            ],
            (
                "PHASE D - ONE CHUNK DELTA"
            )
        )


        embeddings_d = extract_int(
            output_d,
            "Embeddings created"
        )


        new_changed_d = extract_int(
            output_d,
            "New or changed"
        )


        print(
            "Phase D embeddings created:",
            embeddings_d
        )


        print(
            "Phase D new or changed:",
            new_changed_d
        )


        require(
            embeddings_d == 1,
            (
                "Phase D failed: "
                "expected exactly 1 new embedding"
            )
        )


        require(
            new_changed_d == 1,
            (
                "Phase D failed: "
                "expected exactly 1 changed chunk"
            )
        )


        # =================================================
        # PHASE E
        #
        # Run same changed chunks again.
        # Must be fully idempotent.
        # =================================================

        output_e = run_command(
            [
                sys.executable,
                str(
                    TEST_DIR
                    / "incremental_ingest.py"
                )
            ],
            (
                "PHASE E - IDEMPOTENCY CHECK"
            )
        )


        require(
            "NOTHING CHANGED"
            in output_e,
            (
                "Phase E failed: "
                "second identical ingestion "
                "was not idempotent"
            )
        )


        require(
            "Embeddings created: 0"
            in output_e,
            (
                "Phase E failed: "
                "expected zero embeddings"
            )
        )


        # =================================================
        # Production protection check
        # =================================================

        production_after = (
            production_snapshot()
        )


        require(
            production_before
            == production_after,
            (
                "Production Supabase rows changed "
                "during isolated testing"
            )
        )


        # =================================================
        # Success
        # =================================================

        header(
            "ALL INCREMENTAL INDEXING TESTS PASSED"
        )


        print(
            "Phase A"
        )

        print(
            "New test document "
            "-> full processing PASS"
        )

        print()


        print(
            "Phase B"
        )

        print(
            "Unchanged document "
            "-> upstream SKIP PASS"
        )

        print()


        print(
            "Phase C"
        )

        print(
            "Changed PDF hash "
            "-> Docling rerun PASS"
        )

        print(
            "Same extracted chunks "
            "-> embeddings reused PASS"
        )

        print()


        print(
            "Phase D"
        )

        print(
            "One changed chunk "
            "-> exactly one embedding PASS"
        )

        print()


        print(
            "Phase E"
        )

        print(
            "Repeated identical ingestion "
            "-> embeddings 0 PASS"
        )

        print()


        print(
            "Production Supabase rows "
            "-> UNCHANGED PASS"
        )


    finally:

        # =================================================
        # Always remove test rows.
        # Never delete production source.
        # =================================================

        deleted = (
            delete_test_rows()
        )


        print()
        print(
            "Test rows removed:",
            deleted
        )


        # =================================================
        # Keep workspace for debugging if test failed.
        # Delete it on success manually later.
        # =================================================

        print(
            "Test workspace:",
            TEST_DIR
        )


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":

    main()