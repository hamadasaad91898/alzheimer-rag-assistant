import os
import re
import json
import hashlib
import argparse

from collections import Counter, defaultdict, deque
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client


# =========================================================
# Paths and settings
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "chunks.json"
PDF_FILE = BASE_DIR / "neurosci.pdf"
STATE_FILE = BASE_DIR / "incremental_ingest_state.json"

EMBEDDING_DIMENSION = 3072
EMBEDDING_PROVIDER = "azure_openai"
EMBEDDING_VERSION = "1"


# =========================================================
# Environment
# =========================================================

load_dotenv(
    BASE_DIR / ".env"
)

azure_key = os.getenv(
    "AZURE_OPENAI_API_KEY"
)

azure_endpoint = os.getenv(
    "AZURE_OPENAI_ENDPOINT"
)

embedding_model = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
)

supabase_url = os.getenv(
    "SUPABASE_URL"
)

supabase_key = os.getenv(
    "SUPABASE_KEY"
)


required_vars = {
    "AZURE_OPENAI_API_KEY":
        azure_key,

    "AZURE_OPENAI_ENDPOINT":
        azure_endpoint,

    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT":
        embedding_model,

    "SUPABASE_URL":
        supabase_url,

    "SUPABASE_KEY":
        supabase_key,
}


missing = [
    name
    for name, value
    in required_vars.items()
    if not value
]


if missing:

    raise ValueError(
        "Missing environment variables: "
        + ", ".join(
            missing
        )
    )


# =========================================================
# Clients
# =========================================================

base_url = (
    azure_endpoint.rstrip("/")
    + "/openai/v1/"
)


openai_client = OpenAI(
    api_key=azure_key,
    base_url=base_url
)


supabase = create_client(
    supabase_url,
    supabase_key
)


# =========================================================
# Hash helpers
# =========================================================

def normalize_text(text):

    return re.sub(
        r"\s+",
        " ",
        text.strip()
    )


def get_document_hash(
    file_path
):

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            sha256.update(
                block
            )

    return sha256.hexdigest()


def get_chunk_hash(
    content
):

    normalized = normalize_text(
        content
    )

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


# =========================================================
# Embeddings
# =========================================================

def create_embedding(
    text
):

    response = (
        openai_client
        .embeddings
        .create(
            model=embedding_model,
            input=text
        )
    )

    embedding = (
        response
        .data[0]
        .embedding
    )


    if (
        len(
            embedding
        )
        != EMBEDDING_DIMENSION
    ):

        raise ValueError(
            "Unexpected embedding dimension: "
            f"{len(embedding)}. "
            f"Expected {EMBEDDING_DIMENSION}."
        )


    return embedding


def embedding_metadata_matches(
    row
):

    return (

        row.get(
            "embedding_model"
        )
        == embedding_model

        and row.get(
            "embedding_version"
        )
        == EMBEDDING_VERSION

        and row.get(
            "embedding_dimension"
        )
        == EMBEDDING_DIMENSION

        and row.get(
            "embedding_provider"
        )
        == EMBEDDING_PROVIDER
    )


# =========================================================
# Local successful-ingestion state
# =========================================================

def load_state():

    if not STATE_FILE.exists():

        return None


    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(
                f
            )


        if isinstance(
            data,
            dict
        ):

            return data


    except Exception:

        pass


    return None


def save_state(
    source,
    document_hash,
    chunk_count
):

    data = {

        "source":
            source,

        "document_hash":
            document_hash,

        "chunk_count":
            int(
                chunk_count
            ),

        "embedding_model":
            embedding_model,

        "embedding_version":
            EMBEDDING_VERSION,

        "embedding_dimension":
            EMBEDDING_DIMENSION,

        "embedding_provider":
            EMBEDDING_PROVIDER,
    }


    with open(
        STATE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


def state_matches_current(
    state,
    source,
    document_hash
):

    if not state:

        return False


    return (

        state.get(
            "source"
        )
        == source

        and state.get(
            "document_hash"
        )
        == document_hash

        and state.get(
            "embedding_model"
        )
        == embedding_model

        and state.get(
            "embedding_version"
        )
        == EMBEDDING_VERSION

        and state.get(
            "embedding_dimension"
        )
        == EMBEDDING_DIMENSION

        and state.get(
            "embedding_provider"
        )
        == EMBEDDING_PROVIDER

        and isinstance(
            state.get(
                "chunk_count"
            ),
            int
        )

        and state.get(
            "chunk_count"
        )
        > 0
    )


# =========================================================
# Supabase
# =========================================================

DOCUMENT_SELECT_FIELDS = (

    "id,"
    "chunk_id,"
    "content,"
    "section,"
    "source,"
    "pages,"
    "document_hash,"
    "chunk_hash,"
    "embedding_model,"
    "embedding_version,"
    "embedding_dimension,"
    "embedding_provider"
)


def load_existing_rows(
    source
):

    response = (

        supabase
        .table(
            "documents"
        )
        .select(
            DOCUMENT_SELECT_FIELDS
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


def database_matches_state(
    existing_rows,
    state,
    document_hash
):

    if (
        not existing_rows
        or not state
    ):

        return False


    expected_count = state.get(
        "chunk_count"
    )


    if not isinstance(
        expected_count,
        int
    ):

        return False


    if (
        len(
            existing_rows
        )
        != expected_count
    ):

        return False


    for row in existing_rows:

        if (
            row.get(
                "document_hash"
            )
            != document_hash
        ):

            return False


        if not row.get(
            "chunk_hash"
        ):

            return False


        if not embedding_metadata_matches(
            row
        ):

            return False


    return True


# =========================================================
# Document-level precheck
# Run this BEFORE Docling
# =========================================================

def run_precheck():

    if not PDF_FILE.exists():

        raise FileNotFoundError(
            f"{PDF_FILE.name} was not found"
        )


    source = PDF_FILE.name

    document_hash = (
        get_document_hash(
            PDF_FILE
        )
    )


    state = load_state()


    existing_rows = (
        load_existing_rows(
            source
        )
    )


    state_ok = (
        state_matches_current(
            state,
            source,
            document_hash
        )
    )


    db_ok = (
        database_matches_state(
            existing_rows,
            state,
            document_hash
        )
    )


    print(
        "=" * 60
    )

    print(
        "DOCUMENT-LEVEL INCREMENTAL PRECHECK"
    )

    print(
        "=" * 60
    )


    print(
        "Source:",
        source
    )


    print(
        "Document hash:",
        document_hash
    )


    print(
        "Stored state:",
        (
            "FOUND"
            if state
            else "NOT FOUND"
        )
    )


    print(
        "Database rows:",
        len(
            existing_rows
        )
    )


    print()


    # =====================================================
    # Entire document unchanged
    # =====================================================

    if (
        state_ok
        and db_ok
    ):

        print(
            "STATUS: UNCHANGED"
        )

        print()

        print(
            "Docling:    SKIP"
        )

        print(
            "Cleaning:   SKIP"
        )

        print(
            "Chunking:   SKIP"
        )

        print(
            "Embeddings: 0"
        )

        print(
            "Database:   NO CHANGES"
        )


        return True


    # =====================================================
    # Processing required
    # =====================================================

    print(
        "STATUS: PROCESSING REQUIRED"
    )


    if not state:

        print(
            "Reason: no successful ingestion "
            "state exists yet."
        )


    elif (
        state.get(
            "document_hash"
        )
        != document_hash
    ):

        print(
            "Reason: PDF hash changed."
        )


    elif not state_ok:

        print(
            "Reason: embedding/index "
            "configuration changed."
        )


    else:

        print(
            "Reason: Supabase does not match "
            "the last successful state."
        )


    print()

    print(
        "Run:"
    )

    print(
        "Docling -> Cleaning -> Chunking"
    )

    print()

    print(
        "Then run:"
    )

    print(
        "python incremental_ingest.py"
    )


    return False


# =========================================================
# Load and validate chunks
# =========================================================

def load_current_chunks(
    source
):

    if not CHUNKS_FILE.exists():

        raise FileNotFoundError(
            f"{CHUNKS_FILE.name} was not found"
        )


    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(
            f
        )


    if (
        not isinstance(
            chunks,
            list
        )
        or not chunks
    ):

        raise ValueError(
            "chunks.json must contain "
            "a non-empty JSON list"
        )


    required = {
        "chunk_id",
        "content",
        "section",
        "source",
        "pages",
    }


    current_chunks = []


    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        if not isinstance(
            chunk,
            dict
        ):

            raise ValueError(
                f"Chunk #{index} "
                "is not an object"
            )


        missing_fields = [

            field

            for field
            in required

            if field not in chunk
        ]


        if missing_fields:

            raise ValueError(
                f"Chunk #{index} "
                "is missing fields: "
                + ", ".join(
                    missing_fields
                )
            )


        if (
            chunk.get(
                "source"
            )
            != source
        ):

            raise ValueError(
                f"Chunk #{index} "
                "source mismatch: "
                f"{chunk.get('source')} "
                f"!= {source}"
            )


        content = str(
            chunk[
                "content"
            ]
        )


        if not normalize_text(
            content
        ):

            raise ValueError(
                f"Chunk #{index} "
                "has empty content"
            )


        current_chunks.append(
            {
                **chunk,

                "chunk_hash":
                    get_chunk_hash(
                        content
                    )
            }
        )


    return current_chunks


# =========================================================
# Chunk-level delta ingestion
# =========================================================

def run_ingestion():

    if not PDF_FILE.exists():

        raise FileNotFoundError(
            f"{PDF_FILE.name} was not found"
        )


    source = PDF_FILE.name


    document_hash = (
        get_document_hash(
            PDF_FILE
        )
    )


    current_chunks = (
        load_current_chunks(
            source
        )
    )


    existing_rows = (
        load_existing_rows(
            source
        )
    )


    # =====================================================
    # Full idempotency check
    # =====================================================

    current_hashes = Counter(

        chunk[
            "chunk_hash"
        ]

        for chunk
        in current_chunks
    )


    existing_hashes = Counter(

        row[
            "chunk_hash"
        ]

        for row
        in existing_rows

        if row.get(
            "chunk_hash"
        )
    )


    same_document = (

        bool(
            existing_rows
        )

        and all(

            row.get(
                "document_hash"
            )
            == document_hash

            for row
            in existing_rows
        )
    )


    same_chunks = (

        len(
            existing_rows
        )
        == len(
            current_chunks
        )

        and existing_hashes
        == current_hashes
    )


    same_embedding_model = (

        bool(
            existing_rows
        )

        and all(

            embedding_metadata_matches(
                row
            )

            for row
            in existing_rows
        )
    )


    # =====================================================
    # Nothing changed
    # =====================================================

    if (
        same_document
        and same_chunks
        and same_embedding_model
    ):

        # Important:
        # create/update state even if DB was already correct.
        save_state(
            source,
            document_hash,
            len(
                current_chunks
            )
        )


        print(
            "=" * 60
        )

        print(
            "NOTHING CHANGED"
        )

        print(
            "=" * 60
        )


        print(
            "Unchanged:",
            len(
                current_chunks
            )
        )


        print(
            "New or changed: 0"
        )


        print(
            "Deleted: 0"
        )


        print(
            "Embeddings created: 0"
        )


        print(
            "State file:",
            STATE_FILE.name
        )


        return


    # =====================================================
    # Group old rows by chunk hash
    # =====================================================

    old_by_hash = defaultdict(
        deque
    )


    for row in existing_rows:

        chunk_hash = row.get(
            "chunk_hash"
        )


        if chunk_hash:

            old_by_hash[
                chunk_hash
            ].append(
                row
            )


    unchanged_count = 0
    embedded_count = 0
    new_or_changed_count = 0
    deleted_count = 0


    # =====================================================
    # Process current chunks
    # =====================================================

    for chunk in current_chunks:

        chunk_hash = chunk[
            "chunk_hash"
        ]


        old_row = None


        if old_by_hash[
            chunk_hash
        ]:

            old_row = (
                old_by_hash[
                    chunk_hash
                ].popleft()
            )


        base_data = {

            "chunk_id":
                chunk[
                    "chunk_id"
                ],

            "content":
                chunk[
                    "content"
                ],

            "section":
                chunk[
                    "section"
                ],

            "source":
                chunk[
                    "source"
                ],

            "pages":
                chunk[
                    "pages"
                ],

            "document_hash":
                document_hash,

            "chunk_hash":
                chunk_hash,

            "embedding_model":
                embedding_model,

            "embedding_version":
                EMBEDDING_VERSION,

            "embedding_dimension":
                EMBEDDING_DIMENSION,

            "embedding_provider":
                EMBEDDING_PROVIDER,
        }


        # =================================================
        # Same content + same embedding config
        # Reuse old embedding
        # =================================================

        if (
            old_row
            and embedding_metadata_matches(
                old_row
            )
        ):

            (
                supabase
                .table(
                    "documents"
                )
                .update(
                    base_data
                )
                .eq(
                    "id",
                    old_row[
                        "id"
                    ]
                )
                .execute()
            )


            unchanged_count += 1


            continue


        # =================================================
        # New embedding required
        # =================================================

        embedding = (
            create_embedding(
                chunk[
                    "content"
                ]
            )
        )


        base_data[
            "embedding"
        ] = embedding


        # =================================================
        # Same chunk content but embedding config changed
        # =================================================

        if old_row:

            (
                supabase
                .table(
                    "documents"
                )
                .update(
                    base_data
                )
                .eq(
                    "id",
                    old_row[
                        "id"
                    ]
                )
                .execute()
            )


        # =================================================
        # New or changed chunk
        # =================================================

        else:

            (
                supabase
                .table(
                    "documents"
                )
                .insert(
                    base_data
                )
                .execute()
            )


        embedded_count += 1

        new_or_changed_count += 1


    # =====================================================
    # Delete chunks no longer present
    # =====================================================

    for queue in old_by_hash.values():

        while queue:

            old_row = (
                queue.popleft()
            )


            (
                supabase
                .table(
                    "documents"
                )
                .delete()
                .eq(
                    "id",
                    old_row[
                        "id"
                    ]
                )
                .execute()
            )


            deleted_count += 1


    # =====================================================
    # Save successful state
    #
    # Only save AFTER the whole ingestion completed.
    # =====================================================

    save_state(
        source,
        document_hash,
        len(
            current_chunks
        )
    )


    # =====================================================
    # Final summary
    # =====================================================

    print(
        "=" * 60
    )

    print(
        "INCREMENTAL INGESTION COMPLETED"
    )

    print(
        "=" * 60
    )


    print(
        "Total current chunks:",
        len(
            current_chunks
        )
    )


    print(
        "Unchanged:",
        unchanged_count
    )


    print(
        "New or changed:",
        new_or_changed_count
    )


    print(
        "Deleted:",
        deleted_count
    )


    print(
        "Embeddings created:",
        embedded_count
    )


    print()


    print(
        "Document hash:",
        document_hash
    )


    print(
        "Embedding model:",
        embedding_model
    )


    print(
        "State file:",
        STATE_FILE.name
    )


# =========================================================
# CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Document-level precheck + "
            "chunk-level delta ingestion"
        )
    )


    parser.add_argument(
        "--precheck",
        action="store_true",
        help=(
            "Check PDF hash before Docling "
            "and skip upstream processing "
            "when unchanged"
        )
    )


    args = parser.parse_args()


    if args.precheck:

        run_precheck()

    else:

        run_ingestion()


if __name__ == "__main__":

    main()