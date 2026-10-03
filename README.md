# ML Model Monitoring & Automated Retraining System

An end-to-end machine learning monitoring system that detects data drift, prediction drift, and model performance changes in production data. The system evaluates whether a model should be retrained and only promotes a retrained model when it performs better than the current production model.

## 🚀 Project Overview

Machine learning models can lose reliability after deployment when production data changes over time.

This project simulates a production environment and builds an automated monitoring pipeline that:

- Detects data drift
- Detects prediction drift
- Monitors model performance
- Retrains the model using production data
- Compares the current and retrained models
- Decides whether the retrained model should be promoted
- Maintains a model registry
- Provides a Streamlit monitoring dashboard

## 🏗️ System Architecture

```text
Training Data
      ↓
Data Preprocessing
      ↓
Model Training
      ↓
Production Model
      ↓
Production Data
      ↓
Monitoring
 ┌────┼───────────────┐
 ↓    ↓               ↓
Data  Prediction      Performance
Drift Drift           Monitoring
 └────┼───────────────┘
      ↓
Retraining Decision
      ↓
Candidate Model
      ↓
Model Comparison
      ↓
Promotion Decision
      ↓
Model Registry