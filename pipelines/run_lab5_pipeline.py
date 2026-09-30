import subprocess
import sys

def execute_pipeline():
    print("=" * 55)
    print("LAB 5 - PRODUCTION DATA PIPELINE")
    print("=" * 55)

    scripts = [
        "src/validate_data.py",
        "src/preprocess_pipeline.py",
        "src/validate_outputs.py"
    ]

    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} failed.")
            sys.exit(1)

    print("\n[SUCCESS] LAB 5 PRODUCTION PIPELINE FULLY EXECUTED!")

if __name__ == "__main__":
    execute_pipeline()
