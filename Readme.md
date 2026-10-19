# Heart Disease Prediction – End-to-End MLOps Project

## MLOps Experimental Learning Assignment – Assignment 01

**Course:** Machine Learning Operations (MLOps) – AIMLCZG523  
**Project:** End-to-End ML Model Development, CI/CD, and Production Deployment  
**Dataset:** Heart Disease UCI Dataset  
**Student Name:** <YOUR NAME>  
**Student ID:** <YOUR STUDENT ID>  
**GitHub Repository:** <GITHUB REPOSITORY URL>  
**Deployment:** <DEPLOYED API URL OR LOCAL KUBERNETES>  
**Demo Video:** <VIDEO LINK>

---

## 1. Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) pipeline for predicting the presence or risk of heart disease using patient health attributes.

The project covers the complete machine learning lifecycle:

- Data acquisition
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning model development
- Cross-validation and model evaluation
- Experiment tracking using MLflow
- Model packaging
- Automated testing
- CI/CD using GitHub Actions
- REST API development using FastAPI
- Docker containerization
- Kubernetes deployment
- API logging
- Monitoring using Prometheus/Grafana
- Production-style documentation

The objective is to demonstrate how a machine learning model can be developed reproducibly and moved from experimentation to a production-style deployment.

---

## 2. Problem Statement

Heart disease is a major healthcare concern. Patient clinical measurements can be used to develop machine learning models capable of identifying patterns associated with heart disease.

The objective of this project is to build a binary classification model that predicts whether a patient is at risk/presents evidence of heart disease based on available health attributes.

The final selected model is exposed through a REST API that accepts patient information as JSON and returns a prediction together with model confidence/probability.

> This project is developed for academic purposes and is not intended to provide medical diagnosis or replace professional medical advice.

---

## 3. Dataset

### Dataset Name

Heart Disease UCI Dataset

### Source

UCI Machine Learning Repository

Dataset URL:

<ADD EXACT DATASET URL USED>

### Target Variable

The target represents the presence/absence of heart disease.

The exact target transformation used in this project is:

<DESCRIBE YOUR TARGET COLUMN AND HOW YOU CONVERTED IT TO 0/1>

### Features

The dataset contains patient health attributes such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate
- Exercise-induced angina
- ST depression
- Slope
- Number of major vessels
- Thalassemia-related information

The exact feature set used by the model is documented in the preprocessing/training code.

---

## 4. Data Acquisition

The dataset can be obtained using:

```bash
python data/download_data.py
```

If automatic download is not used, download the dataset from the UCI Machine Learning Repository and place it inside:

```text
data/
```

Expected structure:

```text
data/
├── <RAW_DATASET_FILENAME>
└── <CLEANED_DATASET_FILENAME_IF_CREATED>
```

---

## 5. Data Cleaning and Preprocessing

The preprocessing stage performs the required transformations before model training.

The implemented preprocessing includes:

- Missing-value identification and treatment
- Duplicate-data checking
- Target preparation
- Separation of features and target
- Numerical feature processing
- Categorical feature encoding
- Feature scaling where required
- Train/test splitting
- Reproducible random state

A reusable preprocessing pipeline is used so that the same transformations applied during model training are also applied during inference.

This prevents training-serving inconsistencies.

Implementation:

```text
src/preprocess.py
```

<UPDATE THIS SECTION TO MATCH YOUR ACTUAL IMPLEMENTATION.>

---

## 6. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the characteristics of the dataset before model development.

The EDA includes:

- Dataset shape and structure
- Data types
- Descriptive statistics
- Missing-value analysis
- Duplicate analysis
- Target/class distribution
- Numerical feature distributions
- Histograms
- Correlation analysis
- Correlation heatmap
- Relationships between selected clinical features and the target

EDA notebook:

```text
notebooks/01_eda.ipynb
```

### Key EDA Observations

1. <ADD YOUR FIRST ACTUAL OBSERVATION>
2. <ADD YOUR SECOND ACTUAL OBSERVATION>
3. <ADD YOUR THIRD ACTUAL OBSERVATION>
4. <ADD YOUR CLASS-BALANCE OBSERVATION>
5. <ADD IMPORTANT CORRELATION/FINDING>

Do not leave this section generic. The observations above should be based on the actual outputs generated from the dataset.

---

## 7. Feature Engineering

Feature engineering was performed as part of the reproducible machine learning pipeline.

The final pipeline handles the required transformations before passing the data to the classifier.

The approach includes:

- Selection of model input features
- Numerical preprocessing
- Categorical encoding
- Feature scaling where appropriate
- Reusable transformation pipeline

Using a combined preprocessing and model pipeline ensures that inference uses exactly the same transformations as training.

---

## 8. Model Development

At least two classification algorithms were trained and evaluated.

### Model 1 – Logistic Regression

Logistic Regression was selected as a baseline because it provides a simple and interpretable approach to binary classification.

### Model 2 – Random Forest

Random Forest was evaluated as an ensemble model capable of capturing nonlinear relationships and interactions among input features.

### Model Training

Training implementation:

```text
src/train.py
```

Run training using:

```bash
python -m src.train
```

<CHANGE THE COMMAND IF YOUR ACTUAL TRAINING COMMAND IS DIFFERENT.>

---

## 9. Model Evaluation

Models were evaluated using cross-validation and a held-out test set.

The following metrics were considered:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Additional evaluation includes:

- Confusion matrix
- ROC curve

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | <VALUE> | <VALUE> | <VALUE> | <VALUE> | <VALUE> |
| Random Forest | <VALUE> | <VALUE> | <VALUE> | <VALUE> | <VALUE> |

### Final Model Selection

**Selected Model:** `<FINAL MODEL>`

The final model was selected based on:

<EXPLAIN YOUR ACTUAL RESULTS AND WHY THIS MODEL WAS SELECTED.>

For a heart-disease risk classification use case, recall is an important evaluation metric because false-negative predictions may result in potentially at-risk cases being classified as negative. ROC-AUC was also considered to evaluate discrimination across classification thresholds.

---

## 10. Hyperparameter Tuning

Model tuning was performed to identify suitable model configurations.

Parameters investigated included:

### Logistic Regression

<LIST PARAMETERS ACTUALLY TESTED>

### Random Forest

<LIST PARAMETERS ACTUALLY TESTED>

The final hyperparameters were selected based on cross-validation performance rather than test-set accuracy alone.

---

## 11. Experiment Tracking with MLflow

MLflow is integrated into the model-development workflow for experiment tracking.

For each experiment, the pipeline records relevant information including:

- Model type
- Hyperparameters
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Model artifacts
- Evaluation artifacts/plots where applicable

Start the MLflow interface using:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

MLflow makes it possible to compare different model configurations and maintain a reproducible record of experiments.

### Experiment Summary

**Experiment Name:** `<MLFLOW EXPERIMENT NAME>`

**Number of relevant runs:** `<NUMBER>`

**Best Run:** `<RUN/MODEL>`

**Best ROC-AUC:** `<VALUE>`

MLflow screenshots are available under:

```text
screenshots/mlflow/
```

---

## 12. Model Packaging and Reproducibility

The selected model is saved in reusable form after training.

Model location:

```text
models/<MODEL_FILENAME>
```

The saved artifact contains the transformations/model required for inference.

This enables consistent predictions without manually reproducing preprocessing operations.

### Python Dependencies

Dependencies are defined in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 13. Project Structure

```text
heart-disease-mlops/
│
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml
│
├── api/
│   └── main.py
│
├── data/
│   ├── download_data.py
│   └── <dataset files>
│
├── deployment/
│   ├── deployment.yaml
│   └── service.yaml
│
├── models/
│   └── <final model>
│
├── monitoring/
│   └── <monitoring configuration>
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── reports/
│   └── MLOps_Assignment01.pdf
│
├── screenshots/
│   ├── mlflow/
│   ├── cicd/
│   ├── docker/
│   ├── deployment/
│   └── monitoring/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
│
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

The exact structure may vary slightly depending on the final implementation.

---

## 14. Installation and Setup

### Prerequisites

Install:

- Python 3.x
- pip
- Git
- Docker
- Kubernetes/Minikube or Docker Desktop Kubernetes
- kubectl

### Clone Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd heart-disease-mlops
```

### Create Virtual Environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 15. Running EDA

Open:

```text
notebooks/01_eda.ipynb
```

using Jupyter Notebook/JupyterLab or VS Code.

Execute all cells to reproduce the exploratory analysis.

---

## 16. Training the Models

Run:

```bash
python -m src.train
```

The training pipeline performs preprocessing, model training, evaluation, MLflow logging, and final model persistence according to the implementation.

Expected model artifact:

```text
models/<MODEL_FILENAME>
```

---

## 17. Automated Testing

Unit tests are implemented using Pytest.

Tests cover important functionality such as:

- Data/preprocessing behavior
- Expected model behavior
- Model loading
- Prediction output
- Prediction probability/confidence
- API functionality
- Invalid API input where implemented

Run:

```bash
pytest -v
```

All tests must pass before deployment.

---

## 18. CI/CD Pipeline

GitHub Actions is used to automate validation of the project.

Workflow:

```text
.github/workflows/mlops_pipeline.yml
```

The CI/CD pipeline performs:

```text
Code Push / Pull Request
        |
        v
Checkout Repository
        |
        v
Configure Python
        |
        v
Install Dependencies
        |
        v
Lint / Code Quality
        |
        v
Run Unit Tests
        |
        v
Model Training
        |
        v
Artifacts / Logs
```

<ADD DOCKER BUILD HERE IF IT IS PART OF YOUR ACTUAL WORKFLOW.>

The pipeline is configured to fail when required linting, tests, or training steps fail.

Evidence of successful workflow executions is available under:

```text
screenshots/cicd/
```

---

## 19. Model Serving API

The trained model is served through a REST API implemented using FastAPI.

API implementation:

```text
api/main.py
```

Run locally using:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

FastAPI interactive documentation:

```text
http://localhost:8000/docs
```

### Health Endpoint

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

<CHANGE THIS RESPONSE TO MATCH YOUR ACTUAL API.>

### Prediction Endpoint

```http
POST /predict
```

Example input:

```json
{
  "age": 55,
  "sex": 1,
  "cp": 0,
  "trestbps": 140,
  "chol": 250,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.2,
  "slope": 2,
  "ca": 0,
  "thal": 2
}
```

> The JSON schema above must be updated to exactly match the features expected by the final model.

Example output:

```json
{
  "prediction": 1,
  "confidence": 0.82
}
```

`prediction` represents the model's predicted class and `confidence` represents the corresponding prediction probability/confidence.

---

## 20. Docker Containerization

The API is packaged as a Docker container to provide an isolated and reproducible runtime environment.

### Build Docker Image

```bash
docker build -t heart-disease-api .
```

### Run Container

```bash
docker run --rm -p 8000:8000 heart-disease-api
```

Test:

```text
http://localhost:8000/docs
```

The `/predict` endpoint can then be tested using FastAPI Swagger UI, curl, Postman, or another HTTP client.

Docker evidence is available under:

```text
screenshots/docker/
```

---

## 21. Kubernetes Deployment

The Dockerized application is deployed using Kubernetes.

Deployment configuration:

```text
deployment/
├── deployment.yaml
└── service.yaml
```

Apply the manifests:

```bash
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
```

Verify deployment:

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

Expected outcome:

- Deployment is available
- Application pod is in `Running` state
- Service exposes the API
- `/predict` successfully returns predictions

### Accessing the Application

<ADD YOUR ACTUAL MINIKUBE / DOCKER DESKTOP / CLOUD ACCESS COMMAND OR PUBLIC URL>

Deployment screenshots are available under:

```text
screenshots/deployment/
```

---

## 22. Logging

The API implements application/request logging to provide visibility into production behavior.

Operational information logged includes applicable fields such as:

- Request timestamp
- Endpoint
- HTTP status
- Processing/response time
- Prediction outcome
- Application errors

Sensitive patient information should not be unnecessarily written to application logs.

---

## 23. Monitoring

The application implements monitoring using:

**<PROMETHEUS + GRAFANA / YOUR ACTUAL MONITORING SOLUTION>**

Metrics monitored include applicable measures such as:

- Total API requests
- Prediction requests
- Error count
- Request latency
- HTTP response status

If Prometheus is implemented, application metrics can be exposed through:

```http
GET /metrics
```

Prometheus collects these metrics and Grafana is used for visualization.

Monitoring evidence is stored in:

```text
screenshots/monitoring/
```

---

## 24. System Architecture

The overall MLOps architecture is:

```text
               UCI Heart Disease Dataset
                          |
                          v
                  Data Acquisition
                          |
                          v
                 Data Preprocessing
                          |
                          v
                  Feature Pipeline
                          |
              +-----------+-----------+
              |                       |
              v                       v
     Logistic Regression        Random Forest
              |                       |
              +-----------+-----------+
                          |
                          v
                       MLflow
                 Experiment Tracking
                          |
                          v
                  Best Model Selection
                          |
                          v
                    Saved Pipeline
                          |
                          v
                       FastAPI
                          |
                          v
                       Docker
                          |
                          v
                     Kubernetes
                          |
                          v
                    Prediction API
                          |
                  +-------+-------+
                  |               |
                  v               v
                Logs          Prometheus
                                  |
                                  v
                               Grafana


GitHub Repository
       |
       v
GitHub Actions
       |
       +----> Lint
       |
       +----> Tests
       |
       +----> Training
       |
       +----> Build/Artifacts
```

---

## 25. Reproducibility

The project was designed so that the machine learning workflow can be reproduced from a clean environment.

Reproducibility is supported through:

- Version-controlled source code
- Dependency specification
- Fixed random seeds where appropriate
- Reusable preprocessing pipeline
- Automated training
- Automated tests
- MLflow experiment tracking
- Saved model artifact
- Docker containerization
- Kubernetes deployment manifests

A clean environment should be able to install the required dependencies using `requirements.txt` and execute the documented workflow.

---

## 26. Screenshots and Evidence

Execution evidence is stored under:

```text
screenshots/
```

Evidence includes:

- EDA outputs
- Model evaluation
- MLflow experiments
- Unit-test execution
- GitHub Actions
- Docker image/container
- FastAPI prediction
- Kubernetes deployment
- API deployment
- Monitoring dashboard/metrics

---

## 27. Results

The project successfully demonstrates an end-to-end MLOps workflow from raw data to a deployable machine learning API.

### Final Results

**Selected model:** `<MODEL>`

**Accuracy:** `<VALUE>`

**Precision:** `<VALUE>`

**Recall:** `<VALUE>`

**F1-score:** `<VALUE>`

**ROC-AUC:** `<VALUE>`

**Deployment environment:** `<MINIKUBE / AKS / GKE / EKS / DOCKER DESKTOP>`

**API status:** `<WORKING / DEPLOYED>`

**Monitoring:** `<PROMETHEUS + GRAFANA / OTHER>`

---

## 28. Limitations

The project has several limitations:

- The dataset is relatively small compared with production healthcare datasets.
- Model performance depends on the quality and representativeness of the available data.
- Predictions should not be interpreted as medical diagnoses.
- Additional external validation would be required before considering any real clinical application.
- Production healthcare deployment would require additional security, privacy, governance, fairness, compliance, and model-drift controls.

---

## 29. Future Improvements

Potential future improvements include:

- Automated data validation
- Data and model drift detection
- Model registry and promotion stages
- Automated Docker image publishing
- Continuous deployment
- Cloud-native monitoring and alerting
- Authentication and authorization
- HTTPS
- Automated retraining
- Feature store integration
- More extensive hyperparameter optimization

---

## 30. Deliverables

The project submission contains:

- Source code
- Dataset/download instructions
- EDA notebook
- Data preprocessing code
- Model training/evaluation code
- MLflow experiment tracking
- Saved final model
- FastAPI inference service
- Unit tests
- GitHub Actions CI/CD workflow
- Dockerfile
- Kubernetes manifests
- Logging/monitoring implementation
- Screenshots
- Final assignment report
- GitHub repository
- API access instructions
- Demonstration video

---

## 31. Repository and Demo Links

### GitHub Repository

<YOUR GITHUB URL>

### API

<YOUR PUBLIC API URL>

or

<LOCAL KUBERNETES ACCESS INSTRUCTIONS>

### Video Demonstration

<YOUR VIDEO LINK>

### Final Report

```text
reports/MLOps_Assignment01.pdf
```

---

## 32. Conclusion

This project demonstrates the complete lifecycle of a machine learning solution using MLOps practices.

The workflow covers data preparation, exploratory analysis, feature engineering, model experimentation, MLflow tracking, reproducible model packaging, automated testing, CI/CD, REST API serving, Docker containerization, Kubernetes deployment, logging, and monitoring.

The implementation demonstrates how machine learning experimentation can be transformed into a reproducible and deployable production-style system while maintaining traceability through automated pipelines and experiment tracking.