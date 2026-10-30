import re
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

# Each run gets its own file: python baseline/run_baseline.py after_split
run_name = sys.argv[1] if len(sys.argv) > 1 else "run"
out_path = Path(__file__).parent / f"{run_name}.md"
if out_path.exists():
    sys.exit(f"{out_path.name} already exists, pick another run name")

with open(out_path, "w") as f:
    for q in QUERIES:
        start = time.time()
        final = ""
        for final in screen_resumes_stream(q):   # it streams, so keep the last full text
            pass
        # Candidate headers look like "**Name**\n-----------------"
        candidates = re.findall(r"\*\*(.+?)\*\*\n-{5,}", final)
        f.write(f"## {q}\n\nTime: {time.time() - start:.1f}s\n\n")
        f.write(f"Candidates: {', '.join(candidates)}\n\n{final}\n\n")

print(f"Saved {out_path}")
