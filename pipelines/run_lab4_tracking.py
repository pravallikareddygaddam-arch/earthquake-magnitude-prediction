import subprocess
import sys

def execute_pipeline():
    print("=" * 55)
    print("LAB 4 - MLFLOW EXPERIMENT TRACKING")
    print("=" * 55)

    scripts = [
        "src/preprocess.py",
        "src/train_mlflow.py",
        "src/validate_reproducibility.py"
    ]

    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} failed.")
            sys.exit(1)

    print("\n[SUCCESS] LAB 4 PIPELINE FULLY EXECUTED!")

if __name__ == "__main__":
    execute_pipeline()
