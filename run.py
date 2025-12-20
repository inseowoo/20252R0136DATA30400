import subprocess
import sys

def run_pipeline():
    # List your files in the exact order they should run
    pipeline_files = [
        "encode_embedding.py",
        "calculate_similarity.py",
        "core_class_mining.py",
        "classifier_training.py",
        "evaluate_model.py"
    ]

    for file in pipeline_files:
        print(f"--- Starting: {file} ---")
        
        try:
            # shell=False is safer; it executes the file using the current Python interpreter
            result = subprocess.run([sys.executable, file], check=True, capture_output=False)
            print(f"--- Finished: {file} successfully ---\n")
            
        except subprocess.CalledProcessError as e:
            print(f"!!! Error occurred in {file} !!!")
            print(f"Pipeline halted. Return code: {e.returncode}")
            # Exit the entire pipeline if one step fails
            sys.exit(1)

if __name__ == "__main__":
    run_pipeline()