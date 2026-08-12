# Physics-Guided Residual Machine Learning for Geomechanical Stress Profiles

This repository contains the Python implementation associated with the manuscript:

"Physics-Guided Residual Machine Learning for Cross-Well Reproduction of Calibrated Horizontal-Stress Profiles: A Ten-Model Eaton- and Bowers-Conditioned Benchmark"

## Overview

The workflow implements a physics-guided residual machine-learning framework for reproducing calibrated horizontal-stress profiles across wells.

The approach:
- retains analytical geomechanical stress priors,
- predicts calibration residuals using machine learning,
- evaluates models using Leave-One-Well-Out validation.

## Models Included

The benchmark includes:
- Ridge Regression
- KNN
- SVR
- Random Forest
- Extra Trees
- AdaBoost
- XGBoost
- LightGBM
- CatBoost
- MLP Neural Network

## Installation

```bash
pip install -r requirements.txt
```

## Running the Workflow

Full pipeline:

```bash
python run_pipeline_full.py
```

Single blind-well fold:

```bash
python run_single_fold.py
```

Merge outputs:

```bash
python merge_outputs.py
```

## Repository Structure

```
models/
    Individual machine-learning model implementations

run_pipeline_full.py
    Complete benchmark workflow

run_single_fold.py
    Leave-one-well-out execution

merge_outputs.py
    Result aggregation

requirements.txt
    Python dependencies
```

## Data Availability

The original geological dataset is not included because it contains confidential well information. Users should provide their own compatible input dataset.

## Reproducibility

The workflow reproduces the computational methodology described in the associated manuscript.

## License

An open-source license can be selected by the authors before public release.
