import contextlib
import html
import io
import json
import re

import streamlit as st

from rag_chat import (
    ask,
    openai_client,
    chat_model,
)


# =========================================================
# Page Config
# =========================================================

st.set_page_config(
    page_title="Alzheimer's Evidence Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Settings
# =========================================================

MEMORY_MAX_MESSAGES = 8
MEMORY_MAX_MESSAGE_CHARS = 1400
MEMORY_MAX_OUTPUT_TOKENS = 500


# =========================================================
# Session State
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Dark"


# =========================================================
# Theme
# =========================================================

def get_theme():
    if st.session_state.theme_choice == "Light":
        return {
            "background": "#F6F8FC",
            "surface": "#FFFFFF",
            "surface_2": "#EEF2F8",
            "sidebar": "#FFFFFF",
            "text": "#172033",
            "muted": "#667085",
            "border": "#DCE3EC",
            "primary": "#2563EB",
            "primary_soft": "#E8F0FF",
            "chat_user": "#EEF4FF",
            "chat_assistant": "#FFFFFF",
            "input": "#FFFFFF",
            "shadow": "0 8px 24px rgba(15, 23, 42, 0.06)",
        }

    return {
        "background": "#0B1020",
        "surface": "#101729",
        "surface_2": "#171E2F",
        "sidebar": "#111827",
        "text": "#F8FAFC",
        "muted": "#9CA3AF",
        "border": "#293246",
        "primary": "#60A5FA",
        "primary_soft": "#132544",
        "chat_user": "#151D30",
        "chat_assistant": "#0F1627",
        "input": "#171D2A",
        "shadow": "0 8px 24px rgba(0, 0, 0, 0.18)",
    }


def apply_theme():
    theme = get_theme()

    st.markdown(
        f"""
<style>

:root {{
    --app-bg: {theme["background"]};
    --surface: {theme["surface"]};
    --surface-2: {theme["surface_2"]};
    --sidebar: {theme["sidebar"]};
    --text: {theme["text"]};
    --muted: {theme["muted"]};
    --border: {theme["border"]};
    --primary: {theme["primary"]};
    --primary-soft: {theme["primary_soft"]};
    --input-bg: {theme["input"]};
}}


/* ======================================================
   Main App
   ====================================================== */

html,
body,
[data-testid="stAppViewContainer"],
.stApp {{
    background: var(--app-bg) !important;
    color: var(--text) !important;
}}

[data-testid="stMain"] {{
    background: var(--app-bg) !important;
}}

.main .block-container {{
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}}


/* ======================================================
   Header
   ====================================================== */

header[data-testid="stHeader"] {{
    background: var(--app-bg) !important;
}}

[data-testid="stToolbar"] {{
    color: var(--text) !important;
}}


/* ======================================================
   Sidebar
   ====================================================== */

[data-testid="stSidebar"] {{
    background: var(--sidebar) !important;
    border-right: 1px solid var(--border) !important;
}}

[data-testid="stSidebar"] > div {{
    background: var(--sidebar) !important;
}}

[data-testid="stSidebar"] * {{
    color: var(--text);
}}


/* ======================================================
   Text
   ====================================================== */

h1,
h2,
h3,
h4,
h5,
h6,
p,
li,
label,
span {{
    color: var(--text);
}}

.stCaption,
[data-testid="stCaptionContainer"] {{
    color: var(--muted) !important;
}}


/* ======================================================
   App Header
   ====================================================== */

.app-header {{
    margin-bottom: 1.7rem;
}}

.app-title {{
    color: var(--text);
    font-size: 2rem;
    font-weight: 750;
    letter-spacing: -0.03em;
    margin-bottom: 0.35rem;
}}

.app-subtitle {{
    color: var(--muted);
    font-size: 0.95rem;
}}


/* ======================================================
   Sidebar Cards
   ====================================================== */

.status-card {{
    border: 1px solid var(--border);
    background: var(--surface);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: {theme["shadow"]};
}}

.status-label {{
    color: var(--muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 5px;
}}

.status-value {{
    color: var(--text);
    font-size: 0.95rem;
    font-weight: 600;
}}

.memory-active {{
    border: 1px solid var(--border);
    background: var(--primary-soft);
    border-radius: 12px;
    padding: 11px 13px;
    margin-top: 10px;
    margin-bottom: 10px;
    color: var(--muted);
    font-size: 0.83rem;
    line-height: 1.6;
}}


/* ======================================================
   Chat
   ====================================================== */

div[data-testid="stChatMessage"] {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    color: var(--text) !important;
    box-shadow: {theme["shadow"]};
}}

div[data-testid="stChatMessage"] * {{
    color: var(--text);
}}


/* ======================================================
   Chat Input
   ====================================================== */

[data-testid="stBottomBlockContainer"] {{
    background: var(--app-bg) !important;
}}

[data-testid="stChatInput"] {{
    background: var(--input-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
}}

[data-testid="stChatInput"] > div {{
    background: var(--input-bg) !important;
    border-radius: 16px !important;
}}

[data-testid="stChatInput"] textarea {{
    color: var(--text) !important;
    background: var(--input-bg) !important;
    caret-color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
}}

[data-testid="stChatInput"] textarea::placeholder {{
    color: var(--muted) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--muted) !important;
}}

[data-testid="stChatInput"] button {{
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}}

[data-testid="stChatInput"] button svg {{
    fill: var(--text) !important;
    stroke: var(--text) !important;
}}

[data-testid="stBottomBlockContainer"] [data-baseweb="textarea"] {{
    background: var(--input-bg) !important;
}}

[data-testid="stBottomBlockContainer"] [data-baseweb="base-input"] {{
    background: var(--input-bg) !important;
}}


/* ======================================================
   Buttons
   ====================================================== */

.stButton > button {{
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}}

.stButton > button:hover {{
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}}


/* ======================================================
   Radio Theme Selector
   ====================================================== */

div[role="radiogroup"] {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 5px 8px;
}}

div[role="radiogroup"] label {{
    color: var(--text) !important;
}}


/* ======================================================
   Info Box
   ====================================================== */

[data-testid="stAlert"] {{
    background: var(--primary-soft) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}}


/* ======================================================
   Metrics
   ====================================================== */

[data-testid="stMetric"] {{
    background: transparent !important;
}}

[data-testid="stMetricLabel"] {{
    color: var(--muted) !important;
}}

[data-testid="stMetricValue"] {{
    color: var(--text) !important;
}}


/* ======================================================
   Expander
   ====================================================== */

[data-testid="stExpander"] {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}}

[data-testid="stExpander"] summary {{
    color: var(--text) !important;
}}


/* ======================================================
   Source Cards
   ====================================================== */

.source-card {{
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 15px 17px;
    margin-bottom: 12px;
    background: var(--surface-2);
}}

.source-title {{
    color: var(--text);
    font-size: 0.95rem;
    font-weight: 650;
    margin-bottom: 8px;
}}

.source-meta {{
    color: var(--muted);
    font-size: 0.84rem;
    line-height: 1.8;
}}

.source-meta b {{
    color: var(--text);
}}

.similarity-score {{
    color: var(--primary);
    font-weight: 700;
}}


/* ======================================================
   Safety / Memory
   ====================================================== */

.safety-note {{
    border-left: 3px solid var(--primary);
    padding: 10px 14px;
    color: var(--muted);
    font-size: 0.88rem;
    margin-top: 15px;
}}

.context-note {{
    color: var(--muted);
    font-size: 0.78rem;
    margin-top: 8px;
}}


/* ======================================================
   Dividers
   ====================================================== */

hr {{
    border-color: var(--border) !important;
}}


/* ======================================================
   Hide Streamlit Extras
   ====================================================== */

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

</style>
""",
        unsafe_allow_html=True,
    )


apply_theme()


# =========================================================
# Generic Helpers
# =========================================================

def extract_json_object(text):
    text = (text or "").strip()

    try:
        data = json.loads(text)

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if (
        start == -1
        or end == -1
        or end <= start
    ):
        raise ValueError(
            "No valid JSON object found."
        )

    data = json.loads(
        text[start:end + 1]
    )

    if not isinstance(data, dict):
        raise ValueError(
            "JSON response is not an object."
        )

    return data


def extract_section(
    text,
    start_label,
    end_labels,
):
    if not text:
        return ""

    start_pattern = re.escape(
        start_label
    )

    if end_labels:
        end_pattern = "|".join(
            re.escape(label)
            for label in end_labels
        )

        pattern = (
            rf"{start_pattern}\s*"
            rf"(.*?)"
            rf"(?=\n(?:{end_pattern})\s*|\Z)"
        )

    else:
        pattern = (
            rf"{start_pattern}\s*"
            rf"(.*)\Z"
        )

    match = re.search(
        pattern,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not match:
        return ""

    return match.group(1).strip()


# =========================================================
# Conversation Memory
# =========================================================

MEMORY_RESOLVER_PROMPT = """
You are a conversation-context resolver for an Alzheimer's
disease Retrieval-Augmented Generation system.

Your ONLY job is to determine whether the current user
question depends on previous messages.

You are NOT answering the medical question.

You will receive:

1. Recent conversation history.
2. The current user question.

Use previous messages only when necessary to resolve
references in the current question.

Examples include:

- it
- that
- this treatment
- this drug
- what about treatment
- and its symptoms
- طب وعلاجه
- طب اعراضه
- وده بيعمل ايه
- الجرعة بتاعته

Rules:

1. Preserve the current user's intent exactly.

2. Preserve the language of the current question.

3. Do not answer the question.

4. Do not add medical facts.

5. Do not add diagnoses.

6. Do not add treatments.

7. Do not broaden the request.

8. Never convert a patient-specific request into
   a general educational request.

9. Preserve patient-specific diagnosis treatment
   dosage emergency or clinician-advice intent.

10. Conversation history is context only.

11. If the question already stands alone
    return it unchanged.

12. Set used_memory to true only when history
    was actually necessary.

Return valid JSON only:

{
  "standalone_question": "...",
  "used_memory": true
}
""".strip()


def build_memory_context(messages):
    if not messages:
        return ""

    recent_messages = messages[
        -MEMORY_MAX_MESSAGES:
    ]

    lines = []

    for message in recent_messages:

        role = message.get(
            "role",
            "",
        )

        if role == "user":

            content = (
                message.get(
                    "content",
                    "",
                )
                or ""
            ).strip()

            if not content:
                continue

            lines.append(
                "User: "
                + content[
                    :MEMORY_MAX_MESSAGE_CHARS
                ]
            )

        elif role == "assistant":

            parsed = message.get(
                "parsed",
                {},
            )

            answer = (
                parsed.get(
                    "answer",
                    "",
                )
                or ""
            ).strip()

            if not answer:
                continue

            lines.append(
                "Assistant: "
                + answer[
                    :MEMORY_MAX_MESSAGE_CHARS
                ]
            )

    return "\n\n".join(lines)


def resolve_question_with_memory(
    question,
    messages,
):
    question = (
        question
        or ""
    ).strip()

    if not question:
        return {
            "standalone_question": "",
            "used_memory": False,
        }

    history = build_memory_context(
        messages
    )

    if not history:
        return {
            "standalone_question":
                question,

            "used_memory":
                False,
        }

    try:

        response = (
            openai_client
            .responses
            .create(
                model=chat_model,
                instructions=(
                    MEMORY_RESOLVER_PROMPT
                ),
                input=f"""
Recent conversation history:

{history}

Current user question:

{question}
""".strip(),
                max_output_tokens=(
                    MEMORY_MAX_OUTPUT_TOKENS
                ),
            )
        )

        raw_output = (
            response.output_text
            or ""
        ).strip()

        data = extract_json_object(
            raw_output
        )

        standalone_question = data.get(
            "standalone_question"
        )

        used_memory = data.get(
            "used_memory"
        )

        if not isinstance(
            standalone_question,
            str,
        ):
            raise ValueError(
                "Missing standalone_question."
            )

        standalone_question = (
            standalone_question
            .strip()
        )

        if not standalone_question:
            raise ValueError(
                "Empty standalone_question."
            )

        if not isinstance(
            used_memory,
            bool,
        ):
            used_memory = False

        return {
            "standalone_question":
                standalone_question,

            "used_memory":
                used_memory,
        }

    except Exception:

        return {
            "standalone_question":
                question,

            "used_memory":
                False,
        }


# =========================================================
# Run RAG
# =========================================================

def run_rag(question):
    log_buffer = io.StringIO()

    with contextlib.redirect_stdout(
        log_buffer
    ):
        result = ask(
            question
        )

    if result is None:
        raise RuntimeError(
            "The RAG pipeline returned no response."
        )

    return str(result)


# =========================================================
# Citation Parser
# =========================================================

def parse_citation_items(
    citations_text,
):
    if not citations_text:
        return []

    pattern = re.compile(
        r"-\s*Section:\s*(.*?)\s*\n"
        r"\s*Pages:\s*(.*?)\s*\n"
        r"\s*Chunk ID:\s*(.*?)\s*\n"
        r"\s*Source:\s*(.*?)\s*\n"
        r"\s*Retrieval Score:\s*([0-9.]+)",
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    )

    items = []

    for match in pattern.finditer(
        citations_text
    ):

        items.append(
            {
                "section":
                    match.group(1).strip(),

                "pages":
                    match.group(2).strip(),

                "chunk_id":
                    match.group(3).strip(),

                "source":
                    match.group(4).strip(),

                "score":
                    match.group(5).strip(),
            }
        )

    return items


# =========================================================
# RAG Response Parser
# =========================================================

def parse_rag_response(text):

    answer = extract_section(
        text,
        "Answer:",
        [
            "Supporting Evidence:",
            "Citations:",
            "Confidence & Safety:",
            "Safety Note:",
        ],
    )

    evidence = extract_section(
        text,
        "Supporting Evidence:",
        [
            "Citations:",
            "Confidence & Safety:",
            "Safety Note:",
        ],
    )

    citations = extract_section(
        text,
        "Citations:",
        [
            "Confidence & Safety:",
            "Safety Note:",
        ],
    )

    confidence_block = extract_section(
        text,
        "Confidence & Safety:",
        [
            "Safety Note:",
        ],
    )

    safety_note = extract_section(
        text,
        "Safety Note:",
        [],
    )

    def get_field(pattern):
        match = re.search(
            pattern,
            confidence_block,
            flags=re.IGNORECASE,
        )

        if match:
            return (
                match
                .group(1)
                .strip()
            )

        return ""

    if not answer:
        answer = text.strip()

    return {
        "answer":
            answer,

        "evidence":
            evidence,

        "citations":
            citations,

        "citation_items":
            parse_citation_items(
                citations
            ),

        "confidence":
            get_field(
                r"Confidence:\s*(.+)"
            ),

        "citation_coverage":
            get_field(
                r"Citation Coverage:\s*(.+)"
            ),

        "verified_claims":
            get_field(
                r"Verified Claims:\s*(.+)"
            ),

        "verification_rate":
            get_field(
                r"Draft Verification Rate:\s*(.+)"
            ),

        "safety_status":
            get_field(
                r"Safety Status:\s*(.+)"
            ),

        "safety_note":
            safety_note,

        "raw":
            text,
    }


# =========================================================
# Render Metrics
# =========================================================

def render_metrics(parsed):

    values = []

    if parsed["confidence"]:
        values.append(
            (
                "Confidence",
                parsed["confidence"],
            )
        )

    if parsed["citation_coverage"]:
        values.append(
            (
                "Citation Coverage",
                parsed[
                    "citation_coverage"
                ],
            )
        )

    if parsed["verified_claims"]:
        values.append(
            (
                "Verified Claims",
                parsed[
                    "verified_claims"
                ],
            )
        )

    if parsed["verification_rate"]:
        values.append(
            (
                "Verification Rate",
                parsed[
                    "verification_rate"
                ],
            )
        )

    if parsed["safety_status"]:
        values.append(
            (
                "Safety",
                parsed[
                    "safety_status"
                ],
            )
        )

    if not values:
        return

    column_count = min(
        len(values),
        4,
    )

    columns = st.columns(
        column_count
    )

    for index, item in enumerate(
        values
    ):

        label, value = item

        with columns[
            index % column_count
        ]:

            st.metric(
                label,
                value,
            )


# =========================================================
# Render Citation Cards
# =========================================================

def render_citation_cards(
    citation_items,
):
    if not citation_items:
        return

    st.markdown(
        "#### Sources"
    )

    for citation in citation_items:

        section = html.escape(
            citation["section"]
        )

        pages = html.escape(
            citation["pages"]
        )

        chunk_id = html.escape(
            citation["chunk_id"]
        )

        source = html.escape(
            citation["source"]
        )

        score = html.escape(
            citation["score"]
        )

        card = (
            '<div class="source-card">'
            f'<div class="source-title">'
            f'Chunk {chunk_id}'
            '</div>'
            '<div class="source-meta">'
            f'<b>Section:</b> {section}<br>'
            f'<b>Pages:</b> {pages}<br>'
            f'<b>Source:</b> {source}<br>'
            f'<b>Similarity Score:</b> '
            f'<span class="similarity-score">'
            f'{score}'
            '</span>'
            '</div>'
            '</div>'
        )

        st.markdown(
            card,
            unsafe_allow_html=True,
        )


# =========================================================
# Render Sources
# =========================================================

def render_sources(parsed):

    evidence = parsed[
        "evidence"
    ]

    citations = parsed[
        "citations"
    ]

    citation_items = parsed[
        "citation_items"
    ]

    if (
        not evidence
        and not citations
    ):
        return

    with st.expander(
        "View verified sources",
        expanded=False,
    ):

        if evidence:

            st.markdown(
                "#### Supporting Evidence"
            )

            st.markdown(
                evidence
            )

        if citation_items:

            render_citation_cards(
                citation_items
            )

        elif citations:

            st.markdown(
                "#### Citations"
            )

            st.markdown(
                citations
            )


# =========================================================
# Render Assistant
# =========================================================

def render_assistant_response(
    parsed,
    memory_used=False,
    standalone_question="",
):

    st.markdown(
        parsed["answer"]
    )

    if memory_used:

        st.markdown(
            """
<div class="context-note">
Conversation context was used to understand this follow-up question.
</div>
""",
            unsafe_allow_html=True,
        )

        if standalone_question:

            with st.expander(
                "View resolved question",
                expanded=False,
            ):

                st.caption(
                    "Question sent to the RAG after resolving conversation context:"
                )

                st.write(
                    standalone_question
                )

    has_metrics = any(
        [
            parsed["confidence"],
            parsed["citation_coverage"],
            parsed["verified_claims"],
            parsed["verification_rate"],
            parsed["safety_status"],
        ]
    )

    if has_metrics:

        st.divider()

        render_metrics(
            parsed
        )

    render_sources(
        parsed
    )

    if parsed["safety_note"]:

        safety_note = html.escape(
            parsed["safety_note"]
        )

        st.markdown(
            (
                '<div class="safety-note">'
                f'{safety_note}'
                '</div>'
            ),
            unsafe_allow_html=True,
        )


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.markdown(
        "## 🧠 Alzheimer's RAG"
    )

    st.caption(
        "Evidence-grounded medical information"
    )

    st.divider()

    st.markdown(
        "#### Appearance"
    )

    st.radio(
        "Appearance",
        options=[
            "Dark",
            "Light",
        ],
        horizontal=True,
        key="theme_choice",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        (
            '<div class="status-card">'
            '<div class="status-label">'
            'Knowledge Base'
            '</div>'
            '<div class="status-value">'
            "Alzheimer's Medical Source"
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="status-card">'
            '<div class="status-label">'
            'System Status'
            '</div>'
            '<div class="status-value">'
            '🟢 Ready'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="status-card">'
            '<div class="status-label">'
            'Conversation Memory'
            '</div>'
            '<div class="status-value">'
            '🟢 Active'
            '</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.caption(
        "The knowledge base is managed by the system administrator."
    )

    st.caption(
        "Users cannot upload or modify source documents."
    )

    st.markdown(
        """
<div class="memory-active">
Memory is temporary and limited to the current chat session.
Starting a new chat clears the conversation context.
</div>
""",
        unsafe_allow_html=True,
    )

    st.divider()

    if st.button(
        "＋ New Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.markdown(
        "### Safety"
    )

    st.caption(
        "For educational information only."
    )

    st.caption(
        "The assistant does not provide patient-specific diagnosis treatment decisions or medication dosing."
    )


# =========================================================
# Header
# =========================================================

st.markdown(
    """
<div class="app-header">
<div class="app-title">
Alzheimer's Evidence Assistant
</div>
<div class="app-subtitle">
Ask questions grounded in a verified Alzheimer's medical source
</div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# Empty State
# =========================================================

if not st.session_state.messages:

    st.info(
        "Ask an educational question about Alzheimer's disease in Arabic or English."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
**Example**

What are the main risk factors for Alzheimer's disease?
"""
        )

    with col2:

        st.markdown(
            """
**مثال**

ما هي اهم عوامل الخطر لمرض الزهايمر؟
"""
        )


# =========================================================
# Chat History
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        if message["role"] == "user":

            st.markdown(
                message["content"]
            )

        else:

            render_assistant_response(
                parsed=message[
                    "parsed"
                ],

                memory_used=message.get(
                    "memory_used",
                    False,
                ),

                standalone_question=message.get(
                    "standalone_question",
                    "",
                ),
            )


# =========================================================
# Chat Input
# =========================================================

question = st.chat_input(
    "Ask about Alzheimer's disease..."
)


if question:

    question = question.strip()

    if question:

        # Resolve memory BEFORE saving current message
        memory_result = (
            resolve_question_with_memory(
                question=question,
                messages=(
                    st.session_state
                    .messages
                ),
            )
        )

        standalone_question = (
            memory_result[
                "standalone_question"
            ]
        )

        memory_used = (
            memory_result[
                "used_memory"
            ]
        )

        # Save user message
        st.session_state.messages.append(
            {
                "role":
                    "user",

                "content":
                    question,
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        # Assistant
        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching and verifying evidence..."
            ):

                try:

                    response_text = run_rag(
                        standalone_question
                    )

                    parsed = parse_rag_response(
                        response_text
                    )

                    render_assistant_response(
                        parsed=parsed,
                        memory_used=memory_used,
                        standalone_question=(
                            standalone_question
                        ),
                    )

                    st.session_state.messages.append(
                        {
                            "role":
                                "assistant",

                            "content":
                                response_text,

                            "parsed":
                                parsed,

                            "memory_used":
                                memory_used,

                            "standalone_question":
                                standalone_question,
                        }
                    )

                except Exception as error:

                    st.error(
                        "The assistant could not process the request right now."
                    )

                    st.caption(
                        str(error)
                    )
