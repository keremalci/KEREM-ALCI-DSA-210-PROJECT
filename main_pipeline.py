"""
Master Pipeline Script - DSA210 Real Estate Analysis Project
Coordinates the entire workflow from data loading to final analysis.

FIX (vs. the original main_pipeline.ipynb): the original called
`analysis/ml_analysis.py`, but only `analysis/ml_analysis.ipynb` existed in
the repo - so this step always failed with "Script not found". A real
`analysis/ml_analysis.py` now exists (see CHANGELOG.md), so this call
works. `test_results.py` is now also a real, runnable script and is run as
Step 1 so a single command reproduces the whole analysis.
"""

import os
import sys
import subprocess


def get_python_executable():
    """Get the Python executable path - use venv if available."""
    venv_path = os.path.join("venv", "Scripts", "python.exe")  # Windows venv
    venv_path_posix = os.path.join("venv", "bin", "python")  # macOS/Linux venv
    if os.path.exists(venv_path):
        return venv_path
    if os.path.exists(venv_path_posix):
        return venv_path_posix
    return sys.executable


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_script(path, description):
    print_header(description)
    print(f"Executing: {path}\n")

    if not os.path.exists(path):
        print(f" ERROR: Script not found at '{path}'")
        return False

    try:
        python_exe = get_python_executable()
        subprocess.run([python_exe, path], check=True)
        print(f"\n '{path}' completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n ERROR: '{path}' failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n ERROR: An unexpected error occurred: {e}")
        return False


def check_data_exists():
    """Verify that raw data files exist."""
    raw_data_dir = "data/raw"
    if not os.path.exists(raw_data_dir):
        print(f" ERROR: Data directory '{raw_data_dir}' not found!")
        return False

    subdirs = ["emlakjet", "sahibinden", "hepsiemlak"]
    found_data = False

    for subdir in subdirs:
        path = os.path.join(raw_data_dir, subdir)
        if os.path.exists(path):
            files = os.listdir(path)
            data_files = [f for f in files if f.endswith(".xlsx") or f.endswith(".csv")]
            if data_files:
                print(f" Found {len(data_files)} data file(s) in {path}")
                found_data = True
            else:
                print(f" No Excel/CSV files found in {path}")

    return found_data


def main():
    print_header("DSA210 REAL ESTATE ANALYSIS PIPELINE")
    print("Welcome! This script will run the complete analysis workflow.\n")

    print("Checking for raw data files...")
    if not check_data_exists():
        print("\n WARNING: No raw data files detected.")
        print("   Make sure you have data in: data/raw/emlakjet/, data/raw/sahibinden/, etc.")
        response = input("\nContinue anyway? (y/n): ").lower()
        if response != "y":
            print("Exiting...")
            return False

    # Step 1: Hypothesis testing report (fast, no ML dependencies beyond scipy)
    run_script("test_results.py", "STEP 1: Running Hypothesis Testing Report")

    # Step 2: Machine Learning & Deal Finder Analysis
    success = run_script(
        "analysis/ml_analysis.py",
        "STEP 2: Running Machine Learning Analysis & Deal Finder",
    )

    if not success:
        print_header("PIPELINE FAILED")
        print("The ML analysis step failed. Please check the error messages above.")
        return False

    print_header("PIPELINE COMPLETED SUCCESSFULLY")
    print(
        """
Your analysis results are ready:

 OUTPUTS GENERATED:
   +-- merged_analysis_plots.png          (hypothesis testing visuals)
   +-- data/outputs/ml_analysis_results.xlsx
   |     +-- Sheet 1: Model Performance (feature set x model, cross-validated)
   |     +-- Sheet 2: Best Deals (undervalued properties, out-of-fold predictions)
   |     +-- Sheet 3: Clustering Analysis (market segments)
   +-- data/outputs/deal_finder_results.png
   +-- final_model.pkl                    (trained ML model)

NEXT STEPS:
   1. Open 'data/outputs/ml_analysis_results.xlsx' to see the best model,
      undervalued listings, and market segments.
   2. Review 'merged_analysis_plots.png' for the hypothesis test visuals.
   3. See CHANGELOG.md for what changed vs. the original analysis and why.

 For more information, check README.md
"""
    )
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
