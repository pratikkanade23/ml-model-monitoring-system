# ML Model Monitoring & Automated Retraining System

An end-to-end machine learning monitoring system designed to detect
data drift, prediction drift, and model performance degradation in
deployed machine learning models.

## Project Overview

Machine learning models can become less reliable when real-world
data changes after deployment.

This project simulates a production ML environment and provides
automated monitoring for:

- Data Drift
- Prediction Drift
- Model Performance Degradation
- Model Retraining

## System Architecture

Training Data
     |
     v
Data Preprocessing
     |
     v
Model Training
     |
     v
Trained ML Model
     |
     v
Production Data
     |
     v
Monitoring System
     |
     +-------------------+
     |                   |
     v                   v
Data Drift         Prediction Drift
     |                   |
     +---------+---------+
               |
               v
       Performance Monitoring
               |
               v
        Drift / Degradation
               |
          +----+----+
          |         |
         NO        YES
          |         |
          v         v
       Continue   Retrain
                    |
                    v
              New Model
                    |
                    v
              Model Evaluation

## Key Features

- Machine learning model training
- Production data simulation
- Statistical data drift detection
- Kolmogorov-Smirnov (KS) test
- Population Stability Index (PSI)
- Prediction drift monitoring
- Model performance monitoring
- Automated retraining trigger
- Model versioning
- Interactive monitoring dashboard
- REST API
- Docker support

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- XGBoost
- Matplotlib
- Plotly
- Streamlit
- FastAPI
- Git
- GitHub
- Docker

## Project Structure

```text
ML-Model-Monitoring-System/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│
├── models/
│
├── app/
│
├── tests/
│
├── logs/
│
├── .gitignore
├── README.md
└── requirements.txt