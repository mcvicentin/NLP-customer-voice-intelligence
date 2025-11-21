# NLP-customer-voice-intelligence

End-to-end **NLP + Big Data** project to analyze customer feedback at scale using:

- **NLP** (sentiment analysis, topic modeling, summarization)
- **Apache Spark** (PySpark + Spark NLP)
- **Databricks** (jobs, notebooks, MLflow)
- **Cloud deployment** (API to serve models + dashboard for insights)

The goal is to simulate a realistic data science project inside a company:  
Ingest customer reviews from multiple sources, process text with Spark, train NLP models,  
and expose the results through an API and an interactive dashboard.

This repository is designed as a **portfolio project** to showcase skills in:

- Natural Language Processing (NLP)
- Big Data / distributed processing (Spark)
- Databricks & MLflow
- Cloud deployment of ML models
- Reproducible, modular ML code

---

## Overview

A company receivws thousands of free-text reviews/day:

- E-commerce product reviews
- App store reviews
- Social media comments
- Support tickets or complaint portals

The **Customer Voice Intelligence** platform aims to:

1. **Collect & ingest** this text data into a data lake  
2. **Process it using Spark**, cleaning and enriching the text  
3. **Train NLP models** to:
   - Predict **sentiment** (positive / neutral / negative)
   - Extract **topics** (What are people talking about?)
   - Generate **summaries** of large volumes of feedback
4. **Serve models via an API** (FastAPI)
5. **Visualize insights in a dashboard** (Streamlit)

---

## Architecture

```text
            ┌───────────────────────────────────────────┐
            │               Data Sources                 │
            │                  Amazon                   │
            └─────────────────────┬─────────────────────┘
                                  │
                                  v
                     ┌────────────────────────────┐
                     │      Ingestion Layer       │
                     │  (Python scripts / jobs)   │
                     └────────────┬───────────────┘
                                  │
                                  v
               ┌────────────────────────────────────────┐
               │         Data Lake (raw / bronze)       │
               │         e.g. cloud object storage      │
               └────────────────────┬───────────────────┘
                                    │
                                    v
                 ┌─────────────────────────────────┐
                 │       Spark NLP Pipeline        │
                 │  cleaning · tokenization · fe   │
                 └─────────────────┬───────────────┘
                                   │
                                   v
             ┌───────────────────────────────────────────┐
             │   ML Training (Spark / Databricks MLflow) │
             │ sentiment · topics · summarization        │
             └─────────────────────┬─────────────────────┘
                                   │
                                   v
                   ┌─────────────────────────────┐
                   │      Model Registry         │
                   │      (e.g. MLflow)          │
                   └─────────────┬───────────────┘
                                 │
                                 v
        ┌───────────────────────────────────────────────────┐
        │ API Service (FastAPI + Docker, cloud-deployed)   │
        │ + Dashboard (Streamlit)                          │
        └───────────────────────────────────────────────────┘
```

## Repository strcture

```
customer-voice-intelligence/
│
├── README.md
├── requirements.txt
├── environment.yaml
├── .gitignore
├── setup.py
├── Makefile
│
├── data/
│   ├── raw/              # raw data sources 
│   └── processed/        # cleaned / enriched datasets
│   
│
├── notebooks/
│   ├── 01_eda_reviews.ipynb              # exploratory analysis of reviews
│   ├── 02_spark_nlp_pipeline.ipynb       # Spark NLP pipeline prototypes
│   ├── 03_sentiment_modeling.ipynb       # sentiment analysis experiments
│   └── 04_topic_modeling.ipynb           # topic modeling (LDA, BERTopic)   
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_ingestion_cleaning/
│   │   ├── __init__.py
│   │   ├── fetch_amazon_reviews.py
|   |   └── text_cleaning.py
│   │
│   ├── spark_jobs/
│   │   ├── __init__.py
│   │   ├── clean_text_spark.py
│   │   ├── nlp_pipeline_spark.py
│   │   ├── feature_engineering_spark.py
│   │   └── spark_session.py
|   |
│   ├── models/
│   │   ├── __init__.py
|   |   └── sentiment/
│   │   │    ├── __init__.py
│   │   │    ├── sentiment_model.py
│   │   │    ├── traditional_models.py
│   │   │    └── bert_trainer.py
|   |   └── topic/
│   │        ├── __init__.py
│   │        ├── topic_model.py
│   │        └── summarization_model.py
│   │
│   ├── dashboards/
│   │   ├── __init__.py
│   │   └── streamlit_app.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_utils.py
│       └── evaluation.py
│
├── databricks/
│   ├── notebooks/        # Databricks notebooks exports
│   ├── jobs/             # job definitions (json/yaml)
│   └── mlflow/           # guidelines / configs for experiment tracking
│
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   └── k8s/
│       └── (optional) Kubernetes manifests
│
└── tests/
    ├── __init__.py
    └── test_basic.py
```

## Stack
- **Language:** Python 3.10+
- **NLP:** spaCy, Hugging Face Transformers, Spark NLP
- **Processing:** Apache Spark (PySpark)
- **Experiment tracking:** MLflow (via Databricks)
- **API:** FastAPI
- **Dashboard:** Streamlit
- **Cloud:** Databricks + Cloud deployment (e.g. GCP / AWS / Azure)


## Roadmap

This project will be developed following the steps below:

- 1) Data collection & EDA
  - Load a public dataset of product/app reviews
  - Explore distributions, language, basic stats
  - 
- 2) Local NLP prototyping
  - Classic text cleaning (lowercasing, punctuation, stopwords)
  - Train a baseline sentiment classifier

- 3) Spark NLP pipeline
  - Move the preprocessing and feature engineering into a Spark job
  - Run the pipeline on larger volumes of data

- 4) Databricks & MLflow
  - Port Spark jobs to Databricks notebooks
  - Track experiments and models with MLflow

- 5) Advanced NLP
  - Fine-tune a Transformer model for sentiment analysis
  - Topic modeling (LDA / BERTopic)
  - Summarization for aggregated reviews

- 6) API & Dashboard
  - Build a FastAPI service to expose the models
  - Create a Streamlit dashboard consuming the API with interactive visualizations


## Status
1) |v| Repository structure
2) |v| Initial Documentation
3) | | Add sample dataset
4) | | Implement basic EDA notebook
5) | | Implement first NLP preprocessing pipeline
6) | | Add Spark jobs and Databricks integration
7) | | Expose models via API & dashboards


## Author
Marcelo Ciani Vicentin
