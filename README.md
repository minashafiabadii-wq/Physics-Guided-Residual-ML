# Physics-Guided Residual Machine Learning for Cross-Well Stress Profile Reproduction

This repository contains the Python implementation associated with the manuscript:

"Physics-Guided Residual Machine Learning for Cross-Well Reproduction of Calibrated Horizontal-Stress Profiles: A Ten-Model Eaton- and Bowers-Conditioned Benchmark"

## Overview

This project implements a physics-guided residual machine learning workflow for reproducing calibrated horizontal-stress profiles across wells.

The framework combines:
- analytical geomechanical stress priors
- machine-learning residual correction
- cross-well validation

## Machine Learning Models

The repository includes implementations of:

- Ridge Regression
- K-Nearest Neighbors (KNN)
- Support Vector Regression (SVR)
- Random Forest
- Extra Trees
- AdaBoost
- XGBoost
- LightGBM
- CatBoost
- Neural Network models

## Requirements

Python libraries:

- numpy
- pandas
- scikit-learn
- xgboost
- lightgbm
- catboost
- matplotlib

## Usage

Run the individual Python scripts corresponding to each machine-learning model.

The workflow requires the appropriate input dataset described in the associated manuscript.

## Reproducibility

The repository provides the source codes used for the machine-learning benchmark and residual prediction workflow.

## License

The code is provided for academic and research purposes.
