from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import json

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "earthquake-magnitude-api"
CONTAINER = "earthquake-magnitude-api-lab11"
PORT = "8001"

def run(command, check=True):
    result = subprocess.run(command, cwd=ROOT, text=True)
    if check and result.returncode:
        raise SystemExit(f"Command failed: {' '.join(command)}")
    return result

def main():
    required = ["Dockerfile", "requirements-deploy.txt", "api/main.py",
                "models/earthquake_model.pkl", "models/preprocessor.pkl"]
    missing = [name for name in required if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(f"Missing Docker deployment files: {missing}")

    run(["docker", "version"])
    run(["docker", "build", "-t", IMAGE, "."])
    run(["docker", "rm", "-f", CONTAINER], check=False)
    run(["docker", "run", "-d", "--name", CONTAINER, "-p", f"{PORT}:8000", IMAGE])

    try:
        healthy = False
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as response:
                    body = json.loads(response.read().decode("utf-8"))
                    if response.status == 200 and body.get("status") == "healthy":
                        healthy = True
                        print("[SUCCESS] Dockerized API health:", body)
                        break
            except Exception:
                time.sleep(2)
        if not healthy:
            run(["docker", "logs", CONTAINER], check=False)
            raise SystemExit("Docker container did not become healthy.")
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metadata", timeout=5) as response:
            print("[SUCCESS] Dockerized API metadata:", response.read().decode("utf-8"))
    finally:
        run(["docker", "rm", "-f", CONTAINER], check=False)

    print(f"[SUCCESS] Lab 11 Docker build and container smoke test passed. Image: {IMAGE}")

if __name__ == "__main__":
    main()
