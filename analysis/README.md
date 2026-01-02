# Analysis

This folder contains analysis scripts and results for the Kurtköy rental price project.

## Files

- **analyze_data.py** - Main analysis script that:
  - Loads and cleans data from all sources
  - Performs statistical analysis
  - Calculates correlations
  - Generates visualizations
  - Creates processed dataset

- **ANALYSIS_SUMMARY.md** - Comprehensive summary of findings including:
  - Key statistics
  - Correlation analysis
  - Insights and recommendations
  - Student housing recommendations

## Running the Analysis

From this directory:

```bash
python analyze_data.py
```

The script will:
1. Load data from `../data/raw/*/` folders
2. Clean and merge the data
3. Perform statistical analysis
4. Generate visualizations (saved to `../visualizations/`)
5. Save processed data to `../data/processed/`

## Output

- Visualizations: Saved to `../visualizations/`
- Processed data: Saved to `../data/processed/`
- Analysis results: Printed to console

## Dependencies

All required packages are listed in the main `requirements.txt` in the project root.
    