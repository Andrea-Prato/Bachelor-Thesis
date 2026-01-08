# Bachelor-Thesis
This project was originally developed during my Bachelor's studies and later reorganized and documented for portfolio purposes.

## Analysis of Human Memory & Physiological Signals
### Project Overview

This project was developed as part of my Bachelor's thesis at the Università della Svizzera Italiana (USI).
It focuses on analyzing physiological signals acquired from wearable devices to study their relationship with human memory performance.

The main objective of the project was to process raw physiological data, extract meaningful features, and explore whether these features could be used to characterize and predict differences in cognitive performance across participants.

### Repository Structure

```cmd

Bachelor-Thesis/
├── code/
│   ├── analysis.ipynb
│   └── process_data/
│       ├── extract_features.py
│       ├── extract_ISCs.py
│       ├── GSR_analysis.py
│       ├── HRV_analysis.py
│       ├── example.py
│       └── requirements.txt
├── dataset/
│   └── timestamps.csv
└── Schedule.txt
```
### Data Description

Physiological data were collected using wearable devices during memory-related experimental tasks.
The dataset includes time-series signals such as:

+ Galvanic Skin Response (GSR)
+ Heart Rate Variability (HRV)

The _timestamps.csv_ file is used to align physiological signals with experimental events and task segments.

### Methodology

The project follows a modular data analysis pipeline:

1. **Data Preprocessing**
  + Cleaning and synchronization of raw physiological signals
  + Handling of artifacts and segmentation based on experimental timestamps
2. **Feature Extraction**
  + Extraction of relevant statistical and signal-based features from GSR and HRV signals
  + Computation of inter-subject correlation (ISC) metrics to analyze shared physiological responses
3. **Exploratory Analysis**
  + Investigation of relationships between extracted features and memory performance
  + Visualization and inspection of feature distributions and correlations
4. **Modeling and Evaluation**
  + Application of machine learning models to assess the predictive value of extracted features
  + Evaluation using different train–test splits to assess robustness

### Main Files

_analysis.ipynb_ - Main notebook for exploratory analysis, visualization, and modeling.

_extract_features.py_ - Feature extraction routines for physiological signals.

_GSR_analysis.py_ - Processing and analysis of Galvanic Skin Response signals.

_HRV_analysis.py_ - Processing and analysis of Heart Rate Variability signals.

_extract_ISCs.py_ - Computation of inter-subject correlations across physiological signals.

### How to Run

1. Install the required dependencies:
```bash
pip install -r code/process_data/requirements.txt
```
2. Run the analysis notebook:
```bash
jupyter notebook code/analysis.ipynb
```
The notebook executes the main analysis pipeline and produces exploratory plots and results.

### Technologies Used
+ Python
+ Pandas, NumPy
+ SciPy, Scikit-learn
+ Matplotlib
+ Jupyter Notebook

### Notes
This project was developed for academic purposes and aims to demonstrate data preprocessing, feature engineering, exploratory data analysis, and basic machine learning workflows applied to real physiological datasets.

