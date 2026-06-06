import os
from groq import Groq
from agents.vector_store import load_store, search
from dotenv import load_dotenv

load_dotenv()

_client = None

def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client

SYSTEM_PROMPT = """You are a school information assistant for KALNET School.
You ONLY answer questions using the provided school documents below.
If the answer is not found in the documents, you MUST respond with:
"I do not have that information — please contact the school office directly."
Never make up or guess any information."""

def chat_with_bot(question: str, conversation_history: list = []):
    # Retrieve relevant chunks via keyword search
    chunks = search(question, k=4)
    
    context = "\n\n".join([
        f"[Source: {c['metadata'].get('source_file', 'unknown')}]\n{c['text']}"
        for c in chunks
    ])
    
    sources = list(set([
        c['metadata'].get('source_file', 'unknown')
        for c in chunks
        if c['metadata'].get('source_file')
    ]))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n\nContext from school documents:\n{context}"},
        {"role": "user", "content": question}
    ]

    client = get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.1,
        max_tokens=800
    )

    answer = response.choices[0].message.content
    return {"answer": answer, "sources": sources}
    