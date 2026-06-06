import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from agents.vector_store import load_store
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a school information assistant for KALNET School.
You ONLY answer questions using the provided school documents below.
If the answer is not found in the documents, you MUST respond with:
"I do not have that information — please contact the school office directly."
Never make up or guess any information.

Context from school documents:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=SYSTEM_PROMPT,
    input_variables=["context", "question"]
)

# Module-level LLM singleton — instantiated once, not per request
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

def format_docs(docs):
    return "\n\n".join([
        f"[Source: {doc.metadata.get('source_file', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    ])

def chat_with_bot(question: str, conversation_history: list = []):
    llm = get_llm()
    store = load_store()  # now returns singleton, no reload
    retriever = store.as_retriever(search_kwargs={"k": 4})

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    retrieved_docs = retriever.invoke(question)
    sources = list(set([
        doc.metadata.get("source_file", "unknown")
        for doc in retrieved_docs
    ]))

    answer = chain.invoke(question)
    return {"answer": answer, "sources": sources}


