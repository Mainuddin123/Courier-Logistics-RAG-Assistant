import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langsmith import traceable

# ============================================================
# PDF
# ============================================================

from langchain_community.document_loaders import PyPDFLoader

# ============================================================
# CHUNKING
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# GEMINI
# ============================================================

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

# ============================================================
# CHROMADB
# ============================================================

import chromadb

# ============================================================
# BM25
# ============================================================

from rank_bm25 import BM25Okapi

# ============================================================
# FLASHRANK
# ============================================================

from flashrank import (
    Ranker,
    RerankRequest,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Courier & Logistics RAG Assistant",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN APP
       ======================================================== */

    .stApp {
        background: #000000;
        color: #ffffff;
    }

    /* ========================================================
       RAINBOW MAIN TITLE
       ======================================================== */

    .rainbow-title {
        text-align: center;
        font-size: 48px;
        font-weight: 900;
        line-height: 1.15;
        margin-top: 10px;
        margin-bottom: 12px;

        background: linear-gradient(
            90deg,
            #00ff88,
            #00d9ff,
            #008cff,
            #7c3aed,
            #ff00aa,
            #ff6600
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ========================================================
       SUBTITLE
       ======================================================== */

    .app-subtitle {
        text-align: center;
        font-size: 16px;
        font-weight: 500;
        color: #b8b8b8;
        margin-bottom: 30px;
    }

    /* ========================================================
       SECTION TITLE
       ======================================================== */

    .section-title {
        color: #00bfff;
        font-size: 26px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* ========================================================
       SUCCESS BOX
       ======================================================== */

    .success-box {
        background: #111111;
        border-left: 4px solid #00ff88;
        border-radius: 10px;
        padding: 16px 20px;
        color: #ffffff;
        margin: 15px 0;
    }

    /* ========================================================
       INFO BOX
       ======================================================== */

    .info-box {
        background: #111111;
        border-left: 4px solid #00bfff;
        border-radius: 10px;
        padding: 16px 20px;
        color: #ffffff;
        margin: 15px 0;
    }

    /* ========================================================
       SOURCE CARD
       ======================================================== */

    .source-card {
        background: #111111;
        border: 1px solid #333333;
        border-left: 3px solid #00d9ff;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
    }

    .source-title {
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .source-file {
        color: #dddddd;
        margin-bottom: 5px;
    }

    .source-page {
        color: #cccccc;
        margin-bottom: 5px;
    }

    .source-score {
        color: #00d9ff;
        font-size: 14px;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .app-footer {
        text-align: center;
        margin-top: 50px;
        padding: 25px 10px;
        border-top: 1px solid #222222;
        background: #050505;
    }

    .footer-name {
        font-size: 15px;
        color: #cccccc;
    }

    .footer-name span {
        color: #00d9ff;
        font-weight: 700;
    }

    .footer-role {
        font-size: 13px;
        color: #777777;
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PATHS / CONFIGURATION
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

DOCUMENT_DIR = (
    BASE_DIR
    / "data"
    / "documents"
)

CHROMA_DIR = (
    BASE_DIR
    / "vectorstores"
    / "chroma"
)

FLASHRANK_CACHE_DIR = (
    BASE_DIR
    / "flashrank_cache"
)


# ============================================================
# RAG CONFIGURATION
# ============================================================

COLLECTION_NAME = "courier_logistics_policy"

EMBEDDING_MODEL = "models/gemini-embedding-2-preview"

# Gemini models for comparison
LLM_MODELS = [
    "gemini-3.5-flash-lite",  # Current model
    "gemini-3.8-flash",       # Model 2
    "gemini-3.7-flash",       # Model 3
]

LLM_MODEL = LLM_MODELS[0]  # Keep current default

FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

TOP_CHROMA = 5
TOP_BM25 = 5
TOP_HYBRID = 5
TOP_FINAL = 3


# ============================================================
# CHUNKING CONFIGURATION
# ============================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

DOCUMENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHROMA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FLASHRANK_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ENVIRONMENT / API KEY
# ============================================================

load_dotenv(
    dotenv_path=BASE_DIR / ".env",
    override=True,
)

# ============================================================
# LANGSMITH TRACING
# ============================================================

os.environ["LANGSMITH_TRACING"] = os.getenv(
    "LANGSMITH_TRACING",
    "true",
)

os.environ["LANGSMITH_PROJECT"] = os.getenv(
    "LANGSMITH_PROJECT",
    "courier-logistics-rag",
)

# ============================================================
# GOOGLE API KEY
# ============================================================

API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)


# ============================================================
# STREAMLIT CLOUD SECRETS FALLBACK
# ============================================================

if not API_KEY:

    try:

        API_KEY = st.secrets.get(
            "GOOGLE_API_KEY"
        )

    except Exception:

        API_KEY = None


if not API_KEY:

    try:

        API_KEY = st.secrets.get(
            "GEMINI_API_KEY"
        )

    except Exception:

        API_KEY = None

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL APP
       ======================================================== */

    .stApp {
        background: #050505 !important;
        color: #eeeeee !important;
    }

    .main {
        background: #050505 !important;
    }

    .block-container {
        max-width: 1100px !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }


    /* ========================================================
       MAIN TITLE
       ======================================================== */

    .main-title {
        font-size: 42px !important;
        font-weight: 900 !important;
        text-align: center !important;
        margin-bottom: 8px !important;

        background: linear-gradient(
            90deg,
            #00d9ff,
            #7a5cff,
            #ff00cc,
            #00d9ff
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .main-subtitle {
        text-align: center !important;
        color: #999999 !important;
        font-size: 15px !important;
        margin-bottom: 30px !important;
    }


    /* ========================================================
       SECTION TITLE
       ======================================================== */

    .section-title {
        font-size: 22px !important;
        font-weight: 800 !important;
        margin-top: 25px !important;
        margin-bottom: 15px !important;

        background: linear-gradient(
            90deg,
            #00d9ff,
            #7a5cff,
            #ff00cc
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }


    /* ========================================================
       STATUS BOX
       ======================================================== */

    .success-box {
        background: #101a15 !important;
        border: 1px solid #1f8f5f !important;
        border-left: 4px solid #00ff88 !important;
        border-radius: 10px !important;

        color: #eeeeee !important;

        padding: 13px 16px !important;
        margin: 15px 0 !important;

        font-size: 14px !important;
        line-height: 1.6 !important;
    }

    .success-box b {
        color: #00ff88 !important;
    }

    .info-box {
        background: #111111 !important;
        border: 1px solid #333333 !important;
        border-left: 4px solid #00d9ff !important;
        border-radius: 10px !important;

        color: #cccccc !important;

        padding: 13px 16px !important;
        margin: 15px 0 !important;

        font-size: 14px !important;
        line-height: 1.6 !important;
    }

    .info-box b {
        color: #00d9ff !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #080808 !important;
        border-right: 1px solid #222222 !important;
    }

    section[data-testid="stSidebar"] > div {
        background: #080808 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #eeeeee;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }


    /* ========================================================
       SIDEBAR BUTTONS
       ======================================================== */

    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;

        background: #151515 !important;
        color: #eeeeee !important;

        border: 1px solid #333333 !important;
        border-radius: 9px !important;

        transition: all 0.2s ease !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #00d9ff !important;
        color: #00d9ff !important;
    }


    /* ========================================================
       RAG PIPELINE
       ======================================================== */

    .pipeline-card {
        background: #0d0d0d !important;

        border: 1px solid #292929 !important;
        border-radius: 16px !important;

        padding: 18px 14px !important;
        margin-top: 15px !important;

        width: 100% !important;
        box-sizing: border-box !important;
    }

    .pipeline-title {
        font-size: 17px !important;
        font-weight: 800 !important;

        background: linear-gradient(
            90deg,
            #00d9ff,
            #7a5cff,
            #ff00cc
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;

        margin-bottom: 15px !important;
    }

    .pipeline-item {
        background: #151515 !important;

        border: 1px solid #292929 !important;
        border-radius: 8px !important;

        color: #eeeeee !important;

        font-size: 14px !important;
        font-weight: 600 !important;

        text-align: center !important;

        padding: 9px 5px !important;
        margin: 4px 0 !important;

        box-sizing: border-box !important;
    }

    .pipeline-arrow {
        color: #00d9ff !important;

        font-size: 15px !important;
        font-weight: 800 !important;

        text-align: center !important;

        padding: 2px 0 !important;
    }

    /* ========================================================
       CHAT MESSAGES
       ======================================================== */

    
[data-testid="stChatMessage"] {
    background: #101010 !important;
    color: #f5f5f5 !important;
    border: 1px solid #292929 !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
    padding: 12px !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em {
    color: #f5f5f5 !important;
    line-height: 1.7 !important;
}

[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] ol {
    color: #f5f5f5 !important;
}

[data-testid="stChatMessage"] li::marker {
    color: #00d9ff !important;
}


    /* ========================================================
       CHAT INPUT
       ======================================================== */

    [data-testid="stChatInput"] {
        background: #0d0d0d !important;
    }

    [data-testid="stChatInput"] textarea {
        background: #111111 !important;

        color: #eeeeee !important;

        border: 1px solid #333333 !important;
        border-radius: 12px !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: #00d9ff !important;

        box-shadow:
            0 0 0 1px #00d9ff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #777777 !important;
    }

/* ============================================================
   VISIBLE SOURCES EXPANDER
   ============================================================ */

div[data-testid="stExpander"] {
    border: 1px solid #303030 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

div[data-testid="stExpander"] details summary {
    background: #171717 !important;
    color: #00d9ff !important;
    border-bottom: 1px solid #303030 !important;
}

div[data-testid="stExpander"] details summary p,
div[data-testid="stExpander"] details summary span,
div[data-testid="stExpander"] details summary svg {
    color: #00d9ff !important;
    fill: #00d9ff !important;
    stroke: #00d9ff !important;
}

div[data-testid="stExpander"] details summary:hover {
    background: #222222 !important;
}

div[data-testid="stExpander"] details div[data-testid="stExpanderDetails"] {
    background: #0d0d0d !important;
}


    /* ========================================================
       SOURCE CARD
       ======================================================== */

    .source-card {
        background: #151515 !important;

        border: 1px solid #333333 !important;
        border-left: 3px solid #00d9ff !important;

        border-radius: 10px !important;

        padding: 14px 16px !important;
        margin: 10px 0 !important;

        color: #eeeeee !important;

        font-size: 14px !important;
        line-height: 1.6 !important;

        box-sizing: border-box !important;
    }

    .source-title {
        color: #ffffff !important;

        font-size: 14px !important;
        font-weight: 700 !important;

        margin-bottom: 8px !important;
    }

    .source-file {
        color: #00d9ff !important;

        font-size: 14px !important;

        margin: 5px 0 !important;
    }

    .source-page {
        color: #bbbbbb !important;

        font-size: 13px !important;

        margin: 5px 0 !important;
    }

    .source-score {
        color: #00ff88 !important;

        font-size: 13px !important;

        margin: 5px 0 !important;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        background: #0d0d0d !important;

        border: 1px solid #292929 !important;
        border-radius: 12px !important;

        padding: 8px !important;
    }

    [data-testid="stFileUploader"] section {
        background: #0d0d0d !important;
    }


    /* ========================================================
       GENERAL BUTTONS
       ======================================================== */

    .stButton > button {
        background: #151515 !important;

        color: #eeeeee !important;

        border: 1px solid #333333 !important;
        border-radius: 9px !important;

        font-weight: 600 !important;

        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        border-color: #00d9ff !important;

        color: #00d9ff !important;
    }


    /* ========================================================
       SPINNER
       ======================================================== */

    [data-testid="stSpinner"] {
        color: #00d9ff !important;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    [data-testid="stAlert"] {
        background: #101010 !important;

        border-radius: 10px !important;
        border: 1px solid #333333 !important;

        color: #eeeeee !important;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer-name {
        text-align: center !important;

        color: #777777 !important;

        font-size: 13px !important;

        margin-top: 40px !important;
        padding-top: 20px !important;
        padding-bottom: 10px !important;

        border-top: 1px solid #222222 !important;
    }

    .footer-name span {
        background: linear-gradient(
            90deg,
            #00d9ff,
            #7a5cff,
            #ff00cc
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;

        font-weight: 700 !important;
    }


    /* ========================================================
       MOBILE RESPONSIVE
       ======================================================== */

    @media (max-width: 768px) {

        .main-title {
            font-size: 30px !important;
        }

        .main-subtitle {
            font-size: 13px !important;
        }

        .section-title {
            font-size: 19px !important;
        }

        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

    }
    
    .block-container {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
}

/* ============================================================
   STREAMLIT HEADER, SPACING AND CHAT STYLING
   ============================================================ */

/* Keep the header visible */
header[data-testid="stHeader"] {
    display: block !important;
    background: #000000 !important;
}

/* Make header and sidebar toggle visible */
header[data-testid="stHeader"] button,
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarExpandButton"] {
    background-color: #151515 !important;
    color: #00d9ff !important;
    border: 1px solid #00d9ff !important;
    border-radius: 8px !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Color toggle icons */
header[data-testid="stHeader"] button svg,
button[data-testid="stSidebarCollapseButton"] svg,
button[data-testid="stSidebarExpandButton"] svg {
    fill: #00d9ff !important;
    stroke: #00d9ff !important;
}

/* Hide only the decorative strip */
div[data-testid="stDecoration"] {
    display: none !important;
}

/* Main application background */
.stApp,
div[data-testid="stAppViewContainer"] {
    background: #000000 !important;
}

/* Restore spacing so the heading is not cut off */
section[data-testid="stMain"] {
    padding-top: 0 !important;
}

div[data-testid="stMainBlockContainer"],
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
}

/* Keep the title fully visible */
.rainbow-title {
    margin-top: 12px !important;
    line-height: 1.2 !important;
    overflow: visible !important;
}

/* Bottom chat area */
div[data-testid="stBottom"],
div[data-testid="stBottomBlockContainer"],
div[data-testid="stChatInput"],
div[data-testid="stChatInput"] > div {
    background: #000000 !important;
}

div[data-testid="stBottomBlockContainer"] {
    padding-bottom: 1rem !important;
}


/* Sidebar toggle: high visibility */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarExpandButton"],
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: #00d9ff !important;
    border: 2px solid #00d9ff !important;
    border-radius: 8px !important;
    z-index: 9999 !important;
}

[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarExpandButton"] svg,
[data-testid="collapsedControl"] svg {
    color: #000000 !important;
    fill: #000000 !important;
    stroke: #000000 !important;
}

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD PDF DOCUMENTS
# ============================================================

def load_pdfs(uploaded_files):

    documents = []

    # --------------------------------------------------------
    # SAVE AND LOAD EACH PDF
    # --------------------------------------------------------

    for uploaded_file in uploaded_files:

        file_path = DOCUMENT_DIR / uploaded_file.name

        # Save uploaded PDF
        with open(file_path, "wb") as file:

            file.write(
                uploaded_file.getbuffer()
            )

        # ----------------------------------------------------
        # LOAD PDF USING PYPDFLOADER
        # ----------------------------------------------------

        try:

            loader = PyPDFLoader(
                str(file_path)
            )

            pdf_documents = loader.load()

            # ------------------------------------------------
            # ADD FILE NAME TO METADATA
            # ------------------------------------------------

            for document in pdf_documents:

                document.metadata["file_name"] = (
                    uploaded_file.name
                )

                # PyPDFLoader uses 0-based page numbers.
                # Convert to human-readable page number.
                if "page" in document.metadata:

                    document.metadata["page"] = (
                        int(document.metadata["page"]) + 1
                    )

                else:

                    document.metadata["page"] = 1

            documents.extend(
                pdf_documents
            )

        except Exception as error:

            st.warning(
                f"Could not load "
                f"{uploaded_file.name}: {error}"
            )

    return documents

# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    return splitter.split_documents(documents)

# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=API_KEY,
    )


# ============================================================
# GEMINI CHAT MODEL
# ============================================================

@st.cache_resource
def get_llm(model_name=None):
    if not API_KEY:
        raise ValueError("Google API key is missing.")

    return ChatGoogleGenerativeAI(
        model=model_name or LLM_MODEL,
        google_api_key=API_KEY,
        temperature=0,
    )


# ============================================================
# CHROMADB CLIENT
# ============================================================

@st.cache_resource
def get_chroma_client():

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )


def get_collection():

    client = get_chroma_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description":
            "Courier and logistics policy documents"
        },
    )


# ============================================================
# FLASHRANK RANKER
# ============================================================

@st.cache_resource
def get_ranker():

    return Ranker(
        model_name=FLASHRANK_MODEL,
        cache_dir=str(FLASHRANK_CACHE_DIR),
    )


# ============================================================
# BM25 TOKENIZER
# ============================================================

def tokenize(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    return text.split()


# ============================================================
# INDEX DOCUMENTS INTO CHROMADB + BM25
# ============================================================

def index_documents(chunks):
    if not chunks:
        raise ValueError("No document chunks were created.")

    collection = get_collection()
    embeddings = get_embeddings()

    # Prepare one consistent list for both retrievers
    valid_chunks = []

    for chunk in chunks:
        text = chunk.page_content.strip()

        if not text:
            continue

        metadata = dict(chunk.metadata or {})

        # Chroma metadata values must be supported scalar types
        safe_metadata = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                safe_metadata[key] = value
            else:
                safe_metadata[key] = str(value)

        valid_chunks.append({
            "text": text,
            "metadata": safe_metadata,
        })

    if not valid_chunks:
        raise ValueError("No valid text found in the PDF chunks.")

    texts = [item["text"] for item in valid_chunks]
    metadatas = [item["metadata"] for item in valid_chunks]
    ids = [f"chunk_{i}" for i in range(len(valid_chunks))]

    # Generate embeddings before replacing the existing index
    vectors = embeddings.embed_documents(texts)

    # Replace the previous collection contents
    existing = collection.get()

    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=vectors,
        metadatas=metadatas,
    )

    # BM25 uses exactly the same order as valid_chunks
    tokenized_documents = [tokenize(text) for text in texts]

    st.session_state.chunks = valid_chunks
    st.session_state.documents = valid_chunks
    st.session_state.bm25 = (
        BM25Okapi(tokenized_documents)
        if any(tokenized_documents)
        else None
    )
    st.session_state.processed = True


# ============================================================
# RESTORE EXISTING CHROMADB INDEX + BM25
# ============================================================

def restore_index():
    try:
        collection = get_collection()

        result = collection.get(
            include=["documents", "metadatas"]
        )

        texts = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        if not texts:
            st.session_state.processed = False
            st.session_state.documents = []
            st.session_state.chunks = []
            st.session_state.bm25 = None
            return

        restored_chunks = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue

            metadata = (
                metadatas[i]
                if i < len(metadatas) and metadatas[i]
                else {}
            )

            restored_chunks.append({
                "text": text,
                "metadata": metadata,
            })

        if not restored_chunks:
            st.session_state.processed = False
            st.session_state.documents = []
            st.session_state.chunks = []
            st.session_state.bm25 = None
            return

        tokenized_documents = [
            tokenize(item["text"])
            for item in restored_chunks
        ]

        st.session_state.chunks = restored_chunks
        st.session_state.documents = restored_chunks
        st.session_state.bm25 = (
            BM25Okapi(tokenized_documents)
            if any(tokenized_documents)
            else None
        )
        st.session_state.processed = True

    except Exception as error:
        st.session_state.processed = False
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.bm25 = None

        st.warning(
            f"Could not restore the existing index: {error}"
        )
        
# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def contextualize_query(query):
    """Rewrite follow-up questions using recent conversation history."""

    messages = st.session_state.get("messages", [])

    if not messages:
        return query

    history = []

    for message in messages[-6:]:
        role = message.get("role", "")
        content = message.get("content", "")

        if role in ("user", "assistant") and content:
            history.append(f"{role.upper()}: {content}")

    if not history:
        return query

    prompt = f"""
Rewrite the latest question as a standalone question
using conversation history only when needed.

Rules:
- Resolve references such as "it", "that", and "what should I do?"
- Preserve the original intent.
- Do not answer the question.
- Do not invent facts.
- Return only the rewritten question.

Conversation history:
{chr(10).join(history)}

Latest question:
{query}

Standalone question:
"""

    try:
        response = get_llm().invoke(prompt)
        rewritten = str(response.content).strip()

        if rewritten:
            return rewritten

    except Exception:
        pass

    return query

# ============================================================
# HYBRID RETRIEVAL
# ============================================================

@traceable(name="hybrid_retrieve")
def hybrid_retrieve(query):

    if not st.session_state.processed:
        return []

    collection = get_collection()
    embeddings = get_embeddings()

    # --------------------------------------------------------
    # CHROMADB SEMANTIC SEARCH
    # --------------------------------------------------------

    query_embedding = embeddings.embed_query(
        query
    )

    chroma_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_CHROMA,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    semantic_results = []

    documents = chroma_results.get(
        "documents",
        [[]],
    )[0]

    metadatas = chroma_results.get(
        "metadatas",
        [[]],
    )[0]

    distances = chroma_results.get(
        "distances",
        [[]],
    )[0]

    for index, text in enumerate(documents):

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        distance = (
            distances[index]
            if index < len(distances)
            else 0.0
        )

        semantic_results.append(
            {
                "text": text,
                "metadata": metadata or {},
                "score": 1.0 / (
                    1.0 + float(distance)
                ),
            }
        )
 

    # --------------------------------------------------------
    # BM25 KEYWORD SEARCH
    # --------------------------------------------------------

    bm25 = st.session_state.bm25

    bm25_results = []

    if bm25 is not None:


        query_tokens = tokenize(query)

        scores = bm25.get_scores(
            query_tokens
        )

        
        top_indexes = [
            index
            for index in sorted(
                range(len(scores)),
                key=lambda index: scores[index],
                reverse=True,
            )
            if scores[index] > 0
        ][:TOP_BM25]


        chunks = st.session_state.chunks

        for index in top_indexes:

            if index >= len(chunks):
                continue

            chunk = chunks[index]

            # Handle LangChain Document
            if hasattr(
                chunk,
                "page_content",
            ):

                text = chunk.page_content

                metadata = (
                    chunk.metadata or {}
                )

            # Handle dictionary
            else:

                text = chunk.get(
                    "text",
                    "",
                )

                metadata = chunk.get(
                    "metadata",
                    {},
                ) or {}

            if not text:
                continue

            bm25_results.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "score": float(
                        scores[index]
                    ),
                }
            )

    # --------------------------------------------------------
    # COMBINE RESULTS
    # --------------------------------------------------------

    combined = {}

    for item in semantic_results:

        key = (
            item["metadata"].get(
                "file_name",
                item["metadata"].get(
                    "source",
                    "Unknown",
                ),
            ),
            item["metadata"].get(
                "page",
                "Unknown",
            ),
            item["text"],
        )

        combined[key] = item

    for item in bm25_results:

        key = (
            item["metadata"].get(
                "file_name",
                item["metadata"].get(
                    "source",
                    "Unknown",
                ),
            ),
            item["metadata"].get(
                "page",
                "Unknown",
            ),
            item["text"],
        )

        if key in combined:

            # Give a small boost when the
            # document appears in both retrievers.

            combined[key]["score"] += (
                item["score"] * 0.01
            )

        else:

            combined[key] = item

    # --------------------------------------------------------
    # FINAL HYBRID RESULTS
    # --------------------------------------------------------

    results = sorted(
        combined.values(),
        key=lambda item: item.get(
            "score",
            0.0,
        ),
        reverse=True,
    )

    return results[:TOP_HYBRID]


# ============================================================
# FLASHRANK RERANKING
# ============================================================

@traceable(name="flashrank_reranking")
def rerank_results(query, documents):

    if not documents:
        return []

    ranker = get_ranker()

    passages = [
        {
            "id": index,
            "text": item["text"],
            "meta": item.get("metadata", {}) or {},
        }
        for index, item in enumerate(documents)
        if item.get("text", "").strip()
    ]

    if not passages:
        return []

    request = RerankRequest(
        query=query,
        passages=passages,
    )

    ranked_passages = ranker.rerank(request)

    results = []

    for passage in ranked_passages:

        score = float(passage.get("score", 0.0))

        # Exclude extremely low-relevance passages.
        if score < 0.01:
            continue

        results.append({
            "text": passage.get("text", ""),
            "metadata": passage.get("meta", {}) or {},
            "score": score,
        })

        if len(results) >= TOP_FINAL:
            break

    return results

# ============================================================
# GENERATE GEMINI ANSWER
# ============================================================

@traceable(name="generate_answer")
def generate_answer(
    query,
    retrieved_documents,
):

    if not retrieved_documents:
        return (
            "The information was not found "
            "in the policy documents."
        )

    # --------------------------------------------------------
    # BUILD POLICY CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for index, item in enumerate(
        retrieved_documents,
        start=1,
    ):

        metadata = item.get(
            "metadata",
            {},
        ) or {}

        source = metadata.get(
            "file_name",
            metadata.get(
                "source",
                "Unknown",
            ),
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        text = item.get(
            "text",
            "",
        )

        if not text:
            continue

        context_parts.append(
            f"""
SOURCE {index}

File: {source}

Page: {page}

Content:
{text}
"""
        )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if not context_parts:

        return (
            "The information was not found "
            "in the policy documents."
        )

    context = "\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # GEMINI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a Courier and Logistics Policy Assistant.

Answer the user's question ONLY using the supplied
policy context.

Rules:

1. Do not invent information.

2. Do not use outside knowledge.

3. If the answer is not supported by the context,
   say exactly:

   "The information was not found in the policy documents."

4. Give a clear and practical answer.

5. Preserve important policy conditions,
   limits, exceptions and procedures.

6. Do not mention embeddings, ChromaDB,
   BM25 or FlashRank.

7. Do not create unsupported citations.

POLICY CONTEXT
==============

{context}

USER QUESTION
=============

{query}

ANSWER
======
"""

    # --------------------------------------------------------
    # CALL GEMINI
    # --------------------------------------------------------
    
    try:
        llm = get_llm(
            st.session_state.get("selected_llm_model", LLM_MODEL))
        response = llm.invoke(prompt)

    except Exception as error:
        print(
            f"Gemini answer generation failed: "
            f"{type(error).__name__}: {error}")
        return (
            "⚠️ The AI service is temporarily unavailable "
            "due to a usage limit or service issue. "
            "Please try again later. Your policy documents "
            "are still available.")

    # --------------------------------------------------------
    # EXTRACT RESPONSE CONTENT
    # --------------------------------------------------------

    content = response.content

    # Gemini/LangChain may return structured content
    if isinstance(
        content,
        list,
    ):

        text_parts = []

        for part in content:

            if isinstance(
                part,
                dict,
            ):

                if part.get(
                    "type"
                ) == "text":

                    text_parts.append(
                        part.get(
                            "text",
                            "",
                        )
                    )

            else:

                text_parts.append(
                    str(part)
                )

        content = "".join(
            text_parts
        )

    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    return str(
        content
    ).strip()



@traceable(name="gemini_model_comparison")
def test_gemini_model(model_name, question, documents):
    """Test a Gemini model using retrieved policy documents."""

    if not documents:
        return "No relevant policy documents found."

    context_parts = []

    for doc in documents:
        # Your retrieval functions return dictionaries
        if isinstance(doc, dict):
            metadata = doc.get("metadata", {}) or {}
            text = doc.get("text", "")
        else:
            # Also support LangChain Document objects
            metadata = getattr(doc, "metadata", {}) or {}
            text = getattr(doc, "page_content", "")

        if not text:
            continue

        source = metadata.get(
            "file_name",
            metadata.get("source", "Unknown")
        )
        page = metadata.get("page", "Unknown")

        context_parts.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {text}"
        )

    if not context_parts:
        return "No usable policy context was found."

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a Courier and Logistics Policy Assistant.

Answer the question ONLY using the policy context.
Do not invent information.

Policy context:
{context}

Question:
{question}

Answer:
"""

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=API_KEY,
        temperature=0,
    )

    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "")
            if isinstance(part, dict)
            else str(part)
            for part in content
        )

    return str(content).strip()

# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(documents):

    if not documents:
        return

    valid_documents = []

    for item in documents:

        score = item.get("score", 0)

        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        # Only display genuinely relevant sources
        if score > 0:
            valid_documents.append(item)

    # No relevant sources
    if not valid_documents:
        return

    with st.expander("📚 Sources", expanded=True):

        seen = set()
        source_index = 1

        for item in valid_documents:

            metadata = item.get("metadata", {}) or {}

            source = metadata.get(
                "file_name",
                metadata.get(
                    "source",
                    "Unknown"
                )
            )

            page = metadata.get(
                "page",
                "Unknown"
            )

            score = item.get("score", 0)

            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0

            # Avoid duplicate file + page
            key = (
                str(source),
                str(page)
            )

            if key in seen:
                continue

            seen.add(key)

            source_name = os.path.basename(
                str(source)
            )
            
            source_html = f"""<div class="source-card">
<div class="source-title">🌐 Source {source_index}</div>
<div class="source-file">📄 {source_name}</div>
<div class="source-page">📑 Page: {page}</div>
<div class="source-score">🎯 FlashRank Score: {score:.4f}</div>
</div>"""

            st.markdown(
                source_html,
                unsafe_allow_html=True,
            )


            source_index += 1

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # SIDEBAR TITLE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="sidebar-title">
            📚 Knowledge<br>
            Base
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-subtitle">
            Upload courier policy PDFs
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # FILE UPLOADER
    # --------------------------------------------------------

    uploaded_files = st.file_uploader(
        "Upload courier policy PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:

        st.caption(
            f"{len(uploaded_files)} PDF(s) selected."
        )

    st.markdown(
        "<div style='height:6px'></div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # PROCESS BUTTON
    # --------------------------------------------------------

    process_button = st.button(
        "⚙️ Process Documents",
        use_container_width=True,
        type="primary",
    )
        

# ============================================================
# SESSION STATE
# ============================================================

if "processed" not in st.session_state:
    st.session_state.processed = False

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "messages" not in st.session_state:
    st.session_state.messages = []



# ============================================================
# PROCESS UPLOADED DOCUMENTS
# ============================================================

if process_button:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF.")
    elif not API_KEY:
        st.sidebar.error(
            "Google API key is missing. Configure GOOGLE_API_KEY."
        )
    else:
        try:
            st.session_state.processed = False
            st.session_state.documents = []
            st.session_state.chunks = []
            st.session_state.bm25 = None

            with st.spinner(
                "Loading PDFs, creating chunks and building indexes..."
            ):
                documents = load_pdfs(uploaded_files)

                if not documents:
                    raise ValueError(
                        "No readable PDF pages were found."
                    )

                chunks = create_chunks(documents)

                if not chunks:
                    raise ValueError(
                        "No text chunks were created from the PDFs."
                    )

                index_documents(chunks)

            st.sidebar.success(
                f"Successfully indexed "
                f"{len(st.session_state.chunks)} chunks."
            )

            st.rerun()

        except Exception as error:
            st.session_state.processed = False
            st.session_state.documents = []
            st.session_state.chunks = []
            st.session_state.bm25 = None

            st.sidebar.error(
                f"Document processing failed: {error}"
            )


# ============================================================
# RESTORE EXISTING INDEX
# ============================================================

if not st.session_state.processed:

    try:

        restore_index()

    except Exception as error:

        # Do not stop the application if no saved
        # index exists yet.

        pass

# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    """
    <div class="rainbow-title">
        📦 Courier & Logistics RAG
        <br>
        Assistant
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-subtitle">
        Hybrid Retrieval + ChromaDB + BM25 + FlashRank + Gemini
    </div>
    """,
    unsafe_allow_html=True,
)
# ============================================================
# STATUS
# ============================================================

if st.session_state.processed:

    chunk_count = len(
        st.session_state.documents
    )

    st.markdown(
        f"""
        <div class="success-box">
            ✅ Knowledge base ready —
            <b>{chunk_count} chunks indexed.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <div class="info-box">
            👈 Upload your courier policy PDFs
            and click <b>Process Documents</b>
            to begin.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT SECTION
# ============================================================

st.markdown(
    """
    <div class="section-title">
        💬 Ask the Assistant
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            display_sources(
                message["sources"]
            )


# ============================================================
# CHAT INPUT
# ============================================================

selected_model = st.selectbox(
    "Choose Gemini model",
    LLM_MODELS,
    index=0,
    key="selected_llm_model",
)

query = st.chat_input(
    "Ask your courier/logistics question..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if query:

    # --------------------------------------------------------
    # REQUIRE KNOWLEDGE BASE
    # --------------------------------------------------------

    if not st.session_state.processed:

        st.warning(
            "Please upload and process "
            "your policy documents first."
        )

        st.stop()

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            query
        )

    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching policy documents..."
        ):

            try:

                # CONTEXTUALIZE FOLLOW-UP QUESTION
                search_query=contextualize_query(query)
 
                # Hybrid Retrieval 
                hybrid_results = hybrid_retrieve(
                    search_query
                )

                # ============================================
                # FLASHRANK
                # ============================================

                final_results = rerank_results(
                    query,
                    hybrid_results,
                )

                # ============================================
                # GEMINI
                # ============================================

                answer = generate_answer(
                    search_query,
                    final_results,
                )

                # ============================================
                # DISPLAY ANSWER
                # ============================================

                st.markdown(
                    answer
                )

                # ============================================
                # DISPLAY SOURCES
                # ============================================

                if final_results:

                    display_sources(
                        final_results
                    )

                # ============================================
                # SAVE ASSISTANT MESSAGE
                # ============================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": final_results,
                    }
                )

            except Exception as error:

                error_message = (
                    f"Error while answering: "
                    f"{error}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )


