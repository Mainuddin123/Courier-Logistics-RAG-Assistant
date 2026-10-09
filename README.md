# 📦 Courier & Logistics RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers courier and logistics policy questions using uploaded PDF documents. It combines hybrid retrieval, vector search, reranking, and Google Gemini to generate context-grounded answers with source references.

**Live Demo:** https://courier-logistics-rag-assistant-zbkgpntkrhgfb4f6om5ufc.streamlit.app/

**GitHub Repository:** [Courier-Logistics-RAG-Assistant](https://github.com/Mainuddin123/Courier-Logistics-RAG-Assistant)

---

## 🚀 Overview

Courier and logistics policies can be difficult to navigate when information is distributed across multiple documents. This application allows users to ask questions in natural language and receive answers based on the relevant policy documents.

Users can upload courier policy PDFs, process the documents, select a Gemini model, and ask questions about delivery attempts, address changes, shipment tracking, returns, cancellations, and other topics covered by the uploaded policies.

The application retrieves relevant passages and displays supporting source documents, page numbers, and FlashRank relevance scores to improve transparency.

## ✨ Key Features

- **PDF Document Processing:** Upload and process courier and logistics policy documents.
- **Persistent Vector Storage:** Store document embeddings in ChromaDB for semantic retrieval.
- **Hybrid Retrieval:** Combine vector similarity search with BM25 keyword retrieval.
- **FlashRank Reranking:** Reorder retrieved passages by relevance to the user's question.
- **Gemini-Powered Answers:** Generate natural-language responses using selectable Google Gemini models.
- **Source Attribution:** Display source PDF filenames, page numbers, and retrieval relevance scores.
- **Context-Grounded Responses:** Instruct the language model to answer using retrieved policy context and avoid unsupported claims.
- **Graceful Error Handling:** Display a fallback message when the AI service is temporarily unavailable.
- **LangSmith Observability:** Trace retrieval, reranking, and generation steps for debugging and monitoring.
- **Interactive Streamlit UI:** Use a single chat interface with a model selector and a document management sidebar.

## 🏗️ Architecture

```text
                  Courier Policy PDFs
                          |
                          v
                Document Processing
                          |
                          v
                 Text Chunking
                          |
                          v
                Gemini Embeddings
                          |
                          v
                  ChromaDB Store
                          |
User Question ---------->|
       |                  |
       v                  v
  BM25 Retrieval + Vector Retrieval
                 |
                 v
          Hybrid Retrieval
                 |
                 v
           FlashRank Reranking
                 |
                 v
       Relevant Policy Context
                 |
                 v
         Google Gemini LLM
                 |
                 v
        Context-Grounded Answer
                 |
                 v
       Answer + Source References
```

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| LLM | Google Gemini |
| Embeddings | Gemini Embedding Model |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 using `rank-bm25` |
| Semantic Retrieval | Vector similarity search |
| Reranking | FlashRank |
| Document Loading | LangChain PDF loader |
| Text Splitting | Recursive Character Text Splitter |
| Observability | LangSmith |
| Version Control | Git and GitHub |
| Deployment | Streamlit Community Cloud |

## 🔄 Application Workflow

1. **Upload documents:** Select one or more courier and logistics policy PDFs.
2. **Process documents:** Extract PDF text, split it into manageable chunks, generate embeddings, and store them in ChromaDB.
3. **Ask a question:** Enter a natural-language question in the chat interface.
4. **Retrieve context:** Search for relevant passages using semantic vector retrieval and BM25 keyword retrieval.
5. **Rerank results:** Use FlashRank to prioritize the most relevant passages.
6. **Generate an answer:** Send the question and retrieved policy context to the selected Gemini model.
7. **Display sources:** Show the generated answer with source filenames, page numbers, and relevance scores.

## 📚 Example Questions

The following examples demonstrate the types of questions the assistant can answer when the relevant information exists in the uploaded policies.

- How many delivery attempts are made?
- Can I change my delivery address after dispatch?
- How can I track my shipment?
- What should I do if my shipment is delayed?
- What is the return or refund policy?
- What should I do if my parcel is lost or damaged?

The actual answer depends on the content of the uploaded policy documents.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.11 or another Python version supported by the project's dependencies
- Git
- A Google Gemini API key
- A LangSmith API key if tracing is enabled

### 1. Clone the repository

```bash
git clone https://github.com/Mainuddin123/Courier-Logistics-RAG-Assistant.git
cd Courier-Logistics-RAG-Assistant
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root if the application uses environment-based configuration.

```env
GOOGLE_API_KEY=your_google_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=courier-logistics-rag
```

Configure these variables according to the environment-variable names used in your application. If your code uses Streamlit secrets instead of a `.env` file, configure the same values in `.streamlit/secrets.toml` or the deployment platform's secrets settings.

**Security:** Never commit API keys, `.env` files, or other secrets to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit in your browser.

## ☁️ Deployment

The application is deployed using Streamlit Community Cloud.

**Live Application:** [Open the deployed assistant](https://courier-logistics-rag-assistant-zbkgpntkrhgfb4f6om5ufc.streamlit.app/)

To deploy your own version:

1. Push the project to a GitHub repository.
2. Connect the repository to Streamlit Community Cloud.
3. Select `app.py` as the application entry point.
4. Configure the required API keys and secrets.
5. Deploy and verify the application using representative policy questions.

The deployed environment must have all dependencies listed in `requirements.txt` and valid API credentials.

## 📊 Monitoring and Evaluation

LangSmith is used to inspect execution traces and diagnose issues across the RAG pipeline.

Monitoring can include:

- Retrieval and reranking execution.
- Language-model request inputs and outputs.
- Latency and error rates.
- API quota and model-access errors.
- Evaluation results for policy-grounded responses.

Source relevance scores indicate the ranking system's assessment of retrieved passages; they do not guarantee answer correctness. Reliable evaluation should also check whether each response is supported by the original policy text.

## ⚠️ Limitations

- Answer generation depends on Gemini API availability, quotas, and model access.
- An API quota error may prevent an answer from being generated even when relevant documents are retrieved successfully.
- Answer quality depends on the quality and completeness of the uploaded PDFs.
- Incorrect, incomplete, or poorly extracted PDF text can affect retrieval and generation.
- The assistant should not be treated as the final authority when a policy is ambiguous or does not contain the requested information.
- Persistent ChromaDB data and uploaded documents must be managed appropriately in local and cloud environments.

## 🔮 Future Enhancements

- Add automated retrieval and answer-quality evaluation.
- Compare models using consistent test questions and evaluation criteria.
- Improve query rewriting and retrieval optimization.
- Add support for more document formats.
- Introduce configurable retrieval parameters and confidence indicators.
- Expand automated tests for document processing, retrieval, and generation.
- Improve caching and API quota handling.

## 🎯 Project Objectives

- Build a practical RAG application for courier and logistics policies.
- Explore the integration of semantic search and keyword retrieval.
- Improve retrieval quality through reranking.
- Generate context-grounded responses using a large language model.
- Provide transparent source references for retrieved information.
- Apply tracing and monitoring to a deployed Generative AI application.

## 👨‍💻 Author

**SK. Khaja Mainuddin**

B.Tech — Artificial Intelligence and Data Science

GitHub: [Mainuddin123](https://github.com/Mainuddin123)

LinkedIn: [SK. Khaja Mainuddin](https://www.linkedin.com/in/khaja-mainuddin-sk-958436348/)

## 📄 License

No license has been specified for this repository. Unless a license is added, standard copyright restrictions apply.

---

*Built as a practical project exploring Retrieval-Augmented Generation, hybrid search, reranking, and production-oriented Generative AI application development.*
