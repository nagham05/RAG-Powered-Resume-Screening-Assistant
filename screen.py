from langchain_openai import ChatOpenAI
from config import LLM_MODEL, OPENROUTER_API_KEY
from retrieve import retrieve

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1,
)


def build_prompt(candidate, query, chunks):
    context = "\n\n".join(chunks[:2])
    return f"AI HR Assistant: Evaluate {candidate} for: {query}\n\nExcerpts:\n{context}\n\nAnswer YES/NO and justify."


def screen_resumes_stream(query):
    candidate_chunks = retrieve(query)

    accumulated_text = ""
    for candidate, chunks in candidate_chunks.items():
        prompt = build_prompt(candidate, query, chunks)

        accumulated_text += f"**{candidate}**\n-----------------\n**Decision:** "
        yield accumulated_text

        # Streaming the LLM response
        for chunk in llm.stream(prompt):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            accumulated_text += content
            yield accumulated_text

        accumulated_text += "\n\n---\n\n"
        yield accumulated_text
