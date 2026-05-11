# Nova Insurance Group Compliance & Policy Assistant

A standalone Streamlit RAG chatbot that answers employee questions using Nova Insurance Group internal policy PDFs.

The assistant loads local PDF policy documents, splits them into chunks, retrieves matching chunks locally, and uses Groq to answer questions only from retrieved policy excerpts.

## Features

- Streamlit chat interface with user and assistant chat bubbles
- Local PDF loading from `documents/`
- PyMuPDF PDF parsing with page-level metadata
- LangChain recursive text splitting
- Local keyword retrieval from prepared document chunks
- Groq answer generation via `llama-3.1-8b-instant`
- Source citations with PDF filename and page number
- Cached vector store with `st.cache_resource`
- Cached local document chunks with `st.cache_resource`
- Clear setup errors for missing API key or missing PDFs

## Project Structure

```text
Nova_Rag_LLM_Project/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
`-- documents/
    |-- 01_AI_Usage_Policy.pdf
    |-- 02_Data_Privacy_Guideline.pdf
    |-- 03_Customer_Data_Handling_Policy.pdf
    |-- 04_Compliance_Approval_Process.pdf
    `-- 05_EU_AI_Act_Internal_Briefing.pdf
```

## Required PDF Names

`app.py` currently expects these exact filenames in `documents/`:

```text
01_AI_Usage_Policy.pdf
02_Data_Privacy_Guideline.pdf
03_Customer_Data_Handling_Policy.pdf
04_Compliance_Approval_Process.pdf
05_EU_AI_Act_Internal_Briefing.pdf
```

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Set your Groq API key:

```powershell
$env:GROQ_API_KEY="your-api-key"
```

Alternatively, create a local `.env` file:

```text
GROQ_API_KEY=your-api-key
```

The `.env` file is ignored by git.

Optionally choose a different Groq model:

```text
GROQ_MODEL=llama-3.1-8b-instant
```

## Run

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

In the sidebar, click **Load Documents** before asking questions. The app does not call Groq during document loading.

## Quota Notes

This app uses `llama-3.1-8b-instant`, retrieves only the top 3 chunks locally, and limits prompt/output size to reduce Groq usage during demos.

Each user question makes only one Groq API call for answer generation. Document loading and retrieval are local and do not call Groq.

If you see a `RESOURCE_EXHAUSTED`, `rate_limit_exceeded`, or `429` error, your Groq account has hit its quota or rate limit. Wait for the retry window, use another API key/project with quota, or check billing and limits in Groq.

## Demo Questions

Try questions such as:

- May employees enter customer data into public AI tools?
- When does a new AI use case require compliance approval?
- What must be documented when AI-generated content is used internally?

The app will answer only from the retrieved policy excerpts. If the answer is not found in the PDFs, it should say so clearly.

## Notes

- Do not commit real internal policy documents unless your organization explicitly allows it.
- The `.gitignore` excludes PDFs by default to reduce the risk of accidentally committing confidential files.
- This chatbot provides guidance based on internal policy documents and does not constitute legal advice.
