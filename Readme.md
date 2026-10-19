cat > README.md <<'EOF'
# Heart Disease Prediction - MLOps Project

## Course
Machine Learning Operations (MLOps) - AIMLCZG523

## Assignment
Experimental Learning Assignment 01

## Project Objective
This project implements an end-to-end MLOps workflow for predicting the
presence or absence of heart disease using the UCI Heart Disease dataset.

The solution covers data acquisition, exploratory data analysis, feature
engineering, model development, experiment tracking, reproducibility,
automated testing, CI/CD, containerization, Kubernetes deployment,
monitoring, and documentation.

## Dataset
Heart Disease Dataset - UCI Machine Learning Repository  
Dataset ID: 45

The dataset is downloaded programmatically using:

```bash
python src/download_data.py