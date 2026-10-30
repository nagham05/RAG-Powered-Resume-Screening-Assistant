import sys
import time
from pathlib import Path

# Make the project root importable so `from app import ...` works
# no matter which folder the script is run from
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import screen_resumes_stream   # your existing function

QUERIES = [
    "Mechanical Engineering and CAD",
    "Data Science and Python",
    "Project Management",
    "Find candidates with Sensor Integration, System Architecture, CAD Design, MATLAB, Data Analysis, Mechanical Engineering experience",
]

with open(Path(__file__).parent / "before.md", "w") as f:
    for q in QUERIES:
        start = time.time()
        final = ""
        for final in screen_resumes_stream(q):   # it streams, so keep the last full text
            pass
        f.write(f"## {q}\n\nTime: {time.time() - start:.1f}s\n\n{final}\n\n")
