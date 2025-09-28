# Antiviral Target Exploration – Influenza A

Predicting antiviral bioactivity of chemical compounds against Influenza A using ChEMBL data, cheminformatics, and machine-learning.

## Overview

This project retrieves bioactivity data for molecules targeting Influenza A from the ChEMBL API
, builds a structured dataset, and trains a machine-learning model to predict bioactivity (IC₅₀) and potency (pIC₅₀) based on molecular fingerprints.
It also visualizes both molecular structures and the model’s regression performance.

## Pipeline
### Data Acquisition

Queried the ChEMBL API for compounds active against Influenza A.

Extracted molecular structures and associated bioactivity metrics (IC₅₀).

### Data Processing

Loaded the retrieved data into a Pandas DataFrame.

Converted IC₅₀ values to pIC₅₀ (−log₁₀(IC₅₀ [M])) for a normalized potency scale.

Cleaned and filtered records (removing nulls, handling duplicates).

### Feature Engineering

Computed molecular fingerprints (e.g., ECFP4) for each compound using cheminformatics tools such as RDKit.

Combined fingerprints with pIC₅₀ values to create a model-ready dataset.

### Predictive Modeling

Split the dataset into training and test sets.

Trained a Random Forest Regressor (or other ensemble model) to predict pIC₅₀.

Evaluated performance with R² score, MAE, and visual regression plots.

### Visualization

Rendered molecule depictions to inspect structural diversity.
<img width="1200" height="197" alt="image" src="https://github.com/user-attachments/assets/0be47bc6-9593-4eb4-ae89-d793df337e2a" />

Created Actual vs Predicted pIC₅₀ plots to assess model fit.
<img width="2067" height="2068" alt="image" src="https://github.com/user-attachments/assets/6b9dca91-b85d-4724-907c-f80a43de5a16" />


