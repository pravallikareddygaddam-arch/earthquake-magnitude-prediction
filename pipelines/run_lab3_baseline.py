import subprocess
import sys

def run_script(script_path):
    print(f"\n[INFO] Running {script_path}...")
    subprocess.run([sys.executable, script_path], check=True)

def main():
    print("=" * 55)
    print("LAB 3 - BASELINE MLOPS PIPELINE")
    print("=" * 55)
    run_script("src/preprocess.py")
    run_script("src/train.py")
    run_script("src/evaluate.py")
    print("\n[SUCCESS] LAB 3 BASELINE PIPELINE COMPLETED!")

if __name__ == "__main__":
    main()
