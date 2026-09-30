import subprocess
import sys

def execute_pipeline():
    print("=" * 55)
    print("LAB 6 - MODEL REGISTRY & LIFECYCLE")
    print("=" * 55)

    scripts = [
        "src/train_registry.py",
        "src/automate_lifecycle.py",
        "src/generate_registry_report.py"
    ]

    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} failed.")
            sys.exit(1)

    print("\n[SUCCESS] LAB 6 MODEL REGISTRY PIPELINE FULLY EXECUTED!")

if __name__ == "__main__":
    execute_pipeline()
