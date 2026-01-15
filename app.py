

import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from collections import defaultdict
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

# 1. Setup Caching for Speed
set_llm_cache(InMemoryCache())
# 1. Custom CSS to add space between columns
custom_css = """
#left-column {
    padding-right: 40px; /* This creates the space between the columns */
}
#right-column {
    padding-left: 20px;
}
"""
# 2. Load Embeddings and Vector Store
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Ensure this folder exists from your notebook save
vectorstore = FAISS.load_local("faiss_resume_index", huggingface_embeddings, allow_dangerous_deserialization=True)

# 3. Setup LLM
access_token = "****" 
llm_engine = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=access_token,
    temperature=0.1,
)
llm = ChatHuggingFace(llm=llm_engine)

# 4. Logic with the exact structure you requested
def screen_resumes_stream(query):
    # Retrieve docs
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    
    candidate_chunks = defaultdict(list)
    for doc in docs:
        name = doc.metadata.get("candidate", "Unknown")
        candidate_chunks[name].append(doc.page_content)
    
    accumulated_text = ""
    for candidate, chunks in candidate_chunks.items():
        context = "\n\n".join(chunks[:2])
        prompt = f"AI HR Assistant: Evaluate {candidate} for: {query}\n\nExcerpts:\n{context}\n\nAnswer YES/NO and justify."
        
        # Exact structure from your request
        accumulated_text += f"**{candidate}**\n-----------------\n**Decision:** "
        yield accumulated_text
        
        # Streaming the LLM response
        for chunk in llm.stream(prompt):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            accumulated_text += content
            yield accumulated_text
        
        accumulated_text += "\n\n---\n\n"
        yield accumulated_text

# 5. Modern Dashboard Design using Blocks
with gr.Blocks(theme=gr.themes.Soft(),css=custom_css) as demo:
    gr.Markdown("# AI-Powered Resume Screening Assistant")
    gr.Markdown("Provide job requirements to analyze and screen candidate resumes effectively.")    
    with gr.Row():
        # Left Panel: Input area
        with gr.Column(scale=1, elem_id="left-column"):
            gr.Markdown("###  Search Criteria")
            job_query = gr.Textbox(
                label="Enter Job Requirements", 
                placeholder="e.g., Python, AWS, 3 years experience...",
                lines=5
            )
            submit_btn = gr.Button("Analyze Resumes", variant="primary")
            
            gr.Examples(
                examples=["Mechanical Engineering and CAD", "Data Science and Python", "Project Management"],
                inputs=job_query
            )
        
        # Right Panel: Output area
        with gr.Column(scale=2, elem_id="right-column"):
            gr.Markdown("### Screening Analysis")
            # This is where your specific formatted text will stream
            output_markdown = gr.Markdown(value="Waiting for input...", label="Evaluation")

    # Connect the button to the function
    submit_btn.click(fn=screen_resumes_stream, inputs=job_query, outputs=output_markdown)

if __name__ == "__main__":
    demo.launch()