from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

for script in ["src/evaluate.py", "pipelines/07_quality_gate.py"]:
    print(f"\\n[Lab 7] Running {script}")
    result = subprocess.run([sys.executable, script], cwd=ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)
print("[SUCCESS] Lab 7 evaluation and quality gate completed.")
