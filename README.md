# Intelligent Bus Service Risk Classification Using Distributed PySpark Big Data Analytics

Big Data Programming Project (ST5011CEM)

## Project Overview

This project develops a distributed big data analytics pipeline for predicting bus service risk using Apache Spark (PySpark). Data is collected from the UK Bus Open Data Service (BODS) by combining three XML data sources:

- Timetables (TransXChange)
- Service Disruptions (SIRI-SX)
- Vehicle Locations (SIRI-VM)

The project includes data ingestion, preprocessing, feature engineering, exploratory data analysis (EDA), machine learning, database integration, and an interactive Streamlit dashboard.

---

# Technologies Used

- Python 3.10
- Apache Spark (PySpark 3.5.8)
- Java JDK
- MySQL
- Streamlit
- Pandas
- Matplotlib
- Scikit-learn
- XML ElementTree

---

# Project Structure

```
BigDataFinal_coursework/
│
├── Data/
│   ├── raw/                 # Original BODS XML files
│   ├── processed/           # Processed Parquet and CSV files
│   └── results/             # Charts, metrics and model outputs
│
├── notebooks/
│   ├── 01_Data_Collection.ipynb
│   ├── 02_Vehicle_Location_Extraction.ipynb
│   ├── 03_SQL_Analysis.ipynb
│   ├── 04_Data_Preprocessing.ipynb
│   ├── 05_EDA_Visualization.ipynb
│   └── 06_ML_Model.ipynb
│
├── dashboard/
│   └── app.py
│
├── drivers/
│   └── mysql-connector-j-9.2.0.jar
│
└── README.md
```

---

# How to Run

Run the notebooks in the following order:

1. **01_Data_Collection.ipynb**
   - Parses timetable XML files
   - Extracts journey information
   - Saves processed journey data

2. **02_Vehicle_Location_Extraction.ipynb**
   - Parses vehicle location XML files
   - Extracts live vehicle positions

3. **03_SQL_Analysis.ipynb**
   - Integrates datasets
   - Performs Spark SQL analysis
   - Exports processed data

4. **04_Data_Preprocessing.ipynb**
   - Cleans data
   - Performs feature engineering
   - Creates the final machine learning dataset

5. **05_EDA_Visualization.ipynb**
   - Generates exploratory data analysis
   - Produces visualizations and statistical summaries

6. **06_ML_Model.ipynb**
   - Trains Logistic Regression, Decision Tree, and Random Forest models
   - Evaluates model performance
   - Saves final metrics

Restart the kernel between notebooks if required.

---

# Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

Requirements:

- Python 3.10+
- Java JDK
- Apache Spark
- MySQL Server

Update the project path in each notebook:

```python
project_path = r"D:\BigDataFinal_coursework"
```

---

# Dataset

The project uses three datasets from the UK Bus Open Data Service (BODS):

### Timetables (TransXChange XML)

Contains:

- Bus journeys
- Stops
- Departure times
- Routes

### Service Disruptions (SIRI-SX XML)

Contains:

- Active disruptions
- Severity
- Affected operators
- Validity periods

### Vehicle Locations (SIRI-VM XML)

Contains:

- Vehicle positions
- Latitude
- Longitude
- Vehicle references

The datasets are integrated using the common `Operator_ID`.

---

# Processed Data

Intermediate datasets are saved to the `Data/processed/` directory, including:

- `bus_risk_dataset.parquet`
- `vehicle_locations.parquet`
- `integrated_journeys.parquet`
- Machine learning training and testing datasets

CSV versions are also exported for external use.

---

# Database

The processed data is stored in a MySQL relational database containing:

- Operators
- Services
- Journeys

The database uses primary keys and foreign key relationships to maintain data integrity.

---

# Streamlit Dashboard

The interactive dashboard displays:

- Project Overview
- Bus Service Analytics
- Machine Learning Results
- Vehicle Location Map
- About Project

Run the dashboard using:

```bash
streamlit run dashboard.py
```

---

# Machine Learning Models

The following classification models were implemented:

- Logistic Regression
- Decision Tree
- Random Forest

Performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

# Author

**Name:** Rojisha Dangol

**Student ID:** 240364


