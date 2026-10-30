import os
from dotenv import load_dotenv

load_dotenv()

# Settings (committed, same for everyone)
RESUME_DIR = "resumes"
INDEX_DIR = "faiss_resume_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/ministral-8b-2512"
TOP_K = 4

# Secrets (from .env, never committed)
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
