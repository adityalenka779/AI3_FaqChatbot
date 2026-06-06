import json
import os
import fitz
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Only PDF and DOCX files are supported.")

SUMMARY_PROMPT = """You are a document analysis assistant for a school administrator.
Analyse the document text below and respond ONLY with a valid JSON object.
No preamble, no markdown, no explanation, no code blocks — just raw JSON.

Return exactly this structure:
{{
  "title_guess": "your best guess at the document title",
  "summary_bullets": ["bullet 1", "bullet 2", "bullet 3", "bullet 4", "bullet 5"],
  "key_dates": ["date 1 with context", "date 2 with context"],
  "action_required": false,
  "action_description": null
}}

Rules:
- summary_bullets must have 5 to 7 items
- key_dates should list every specific date or deadline mentioned
- action_required is true if the document requires the school to do something
- action_description explains what action is needed, or null if none

Document text:
{text}"""

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
        )
    return _llm

def summarise_document(file_path: str) -> dict:
    text = extract_text(file_path)
    word_count = len(text.split())

    if word_count > 3000:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200
        )
        chunks = splitter.split_text(text)
        llm = get_llm()
        chunk_summaries = []
        for chunk in chunks:
            response = llm.invoke(
                f"Summarise this section of a school document in 3 sentences:\n{chunk}"
            )
            chunk_summaries.append(response.content)
        combined = "\n".join(chunk_summaries)
    else:
        combined = text

    llm = get_llm()
    prompt = SUMMARY_PROMPT.format(text=combined)
    response = llm.invoke(prompt)

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)

