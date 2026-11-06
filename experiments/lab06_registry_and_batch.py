from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

for script in ["pipelines/run_lab6_registry.py", "pipelines/run_lab6_batch_inference.py"]:
    print(f"\\n[Lab 6] Running {script}")
    result = subprocess.run([sys.executable, script], cwd=ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)
print("[SUCCESS] Lab 6 registry and batch inference completed.")
