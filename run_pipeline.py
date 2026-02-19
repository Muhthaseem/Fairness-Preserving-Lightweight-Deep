
"""
Pipeline Runner (Automation)
=====================================
Executes the remaining steps of the Deepfake Detection workflow sequentially.
"""
import subprocess
import sys
import time
import os

# Define the pipeline steps (script_path, description)
PIPELINE_STEPS = [
    # ("scripts/03_annotate_demographics.py --batch_size 1024", "STEP 3: Demographic Annotation (FairFace - 32GB VRAM)"),
    # ("scripts/04_create_splits.py",          "STEP 4: Create Train/Val/Test Splits"),
    ("scripts/05_train_teacher.py --batch_size 64",          "STEP 5: Train Teacher (Xception - 16GB VRAM)"),
    ("scripts/06_train_baseline_student.py --batch_size 128", "STEP 6: Train Baseline (MobileNet - 16GB VRAM)"),
    ("scripts/07_train_fair_student.py --batch_size 128",     "STEP 7: Train Fair Student (Distillation - 16GB VRAM)"),
    ("scripts/08_evaluate_all.py",           "STEP 8: Evaluate & Generate Report"),
]

def run_pipeline():
    print("="*80)
    print("   DEEPFAKE DETECTION: AUTOMATED PIPELINE RUNNER")
    print("="*80)
    print(f"System: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Directory: {os.getcwd()}")
    print("-" * 80)

    total_start = time.time()

    for script_name, description in PIPELINE_STEPS:
        print(f"\n\n>>> STARTING: {description}")
        print(f">>> Command:  python {script_name}")
        print("="*80)
        
        step_start = time.time()
        
        # Flush stdout so logs appear immediately
        sys.stdout.flush()
        
        try:
            # Run the script and stream output to the console
            # Split command arguments if present
            cmd_args = [sys.executable] + script_name.split()
            subprocess.run(cmd_args, check=True)
            
            duration = time.time() - step_start
            print(f"\n✅ COMPLETED: {description} (Time: {duration:.1f}s)")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ FAILED: {description}")
            print(f"   Exit Code: {e.returncode}")
            print("\n⛔ Pipeline stopped due to error. Please fix the issue and restart.")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ ERROR executing {script_name}: {e}")
            sys.exit(1)

    total_time = time.time() - total_start
    print(f"\n\n{'='*80}")
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"   Total Time: {total_time/60:.1f} minutes")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_pipeline()
