# PJM AEP Load Anomaly Detection

Anomaly detection on hourly electricity load data from **AEP (American Electric Power)**, part of the PJM Interconnection region, using **Gaussian Mixture Models (GMM)** with **AIC/BIC**-based model selection, compared against a **Linear Regression** baseline.

## 📌 Overview

This project analyzes the `AEP_hourly.csv` time series (hourly megawatt load, 2004–2018) to:

- Engineer time-based features (`hour`, `month`, `day of week`, `year`) from the datetime index
- Fit **Gaussian Mixture Models** on scaled load data and select the optimal number of components using **AIC** (Akaike Information Criterion) and **BIC** (Bayesian Information Criterion)
- Flag anomalous readings using GMM log-likelihood scores (values below the 5th percentile are treated as anomalies)
- Visualize daily anomalies per year in a grid of subplots
- Compare model complexity/fit against a simple **Linear Regression** trend model using AIC/BIC

## 📂 Repository Structure

```
.
├── pjm.ipynb          # Main analysis notebook (feature engineering, GMM fitting, anomaly detection, plots)
├── aic_bic.py         # Standalone script: computes AIC/BIC for GMM vs. Linear Regression on daily-resampled data
├── AEP_hourly.csv      # Hourly AEP load data (Datetime, AEP_MW)
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Dataset

| Column     | Description                          |
|------------|---------------------------------------|
| `Datetime` | Timestamp of the hourly reading       |
| `AEP_MW`   | AEP electricity load in megawatts (MW)|

Source: [PJM Hourly Energy Consumption Data (Kaggle)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

## ⚙️ Methodology

1. **Data loading & cleaning** – parse `Datetime`, set as index, sort chronologically, handle missing files gracefully with fallback demo data.
2. **Feature engineering** – extract `hour`, `month`, `dayofweek`, `year` from the datetime index.
3. **Scaling** – standardize features with `StandardScaler` before modeling.
4. **Model selection (GMM)** – fit GMMs for a range of component counts, compute AIC/BIC for each, and pick the number of components that minimizes BIC.
5. **Anomaly scoring** – compute log-likelihood of each point under the fitted GMM; points below the 5th percentile are flagged as anomalies.
6. **Visualization** – per-year scatter plots of load and log-likelihood with anomalies highlighted, plus daily anomaly counts.
7. **Baseline comparison** – `aic_bic.py` resamples the data to daily means and computes AIC/BIC for both a 2-component GMM and a simple linear trend regression, to compare model fit.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook / JupyterLab

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Usage

**Run the notebook:**

```bash
jupyter notebook pjm.ipynb
```

**Run the standalone AIC/BIC comparison script:**

```bash
python aic_bic.py
```

Make sure `AEP_hourly.csv` is in the same directory as the script/notebook, or update the file path accordingly.

## 📈 Example Output

- AIC/BIC curves across GMM component counts (used to pick the optimal number of components)
- Scatter plots of `AEP_MW` vs. day-of-year, colored by anomaly status, faceted per year
- Log-likelihood plots with the anomaly threshold line
- Daily anomaly count bar/vline plots per year
- Printed AIC/BIC values comparing the GMM model against a linear regression trend model

## 🛠️ Tech Stack

- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) – data handling
- [scikit-learn](https://scikit-learn.org/) – `GaussianMixture`, `LinearRegression`, `StandardScaler`
- [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) – visualization

## 📄 License

This project is licensed under the MIT License — feel free to use and adapt it.

## 🙋 Notes

- The anomaly threshold (5th percentile of log-likelihood) is configurable — adjust `anomaly_threshold_percentile` in the notebook to make detection more or less sensitive.
- For large datasets, consider resampling (e.g., daily means, as done in `aic_bic.py`) to speed up model fitting.
