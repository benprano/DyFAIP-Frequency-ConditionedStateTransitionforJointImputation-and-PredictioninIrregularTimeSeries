# DyFAIP: Frequency-Conditioned State Transition for Joint Imputation-and-Prediction in Irregular Time Series

Official PyTorch implementation of **DyFAIP**, a dual-path framework for joint time-series imputation and downstream tasks on irregular multivariate time series.

---

## 📌 Overview

Irregularly sampled multivariate time series pose significant challenges due to missing values and complex temporal dynamics. **DyFAIP** addresses these issues through a dual-path architecture leveraging **frequency-driven gating mechanisms** and hidden representations. By dynamically conditioning state transitions in the frequency domain, DyFAIP effectively handles missingness patterns while simultaneously optimizing for downstream prediction tasks.

### Key Highlights

* **Frequency-Conditioned State Transitions:** Dynamically gates representations using spectral properties to preserve continuous-time dynamics.
* **Joint Imputation & Prediction:** Co-optimizes missing-data reconstruction and downstream target prediction for mutual performance enhancement.
* **Multi-Domain Benchmarking:** Includes pre-configured pipelines for healthcare (PhysioNet), environmental (Beijing Air Quality), and energy (ETDataset) benchmarks.

---

## 📁 Repository Structure

```text
DyFAIP/
├── datasets/                 # Processed datasets and benchmark files
│   ├── AIR_QUALITY/          # Beijing Air Quality dataset
│   └── ETT-small/            # Electricity Transformer Temperature dataset
├── helpers/                  # Core execution and training logic
│   ├── Runner.py             # Main entry point for training & evaluation
│   ├── metrics.py            # Imputation and prediction evaluation metrics
│   └── trainer_helper.py     # Optimization loops and training utilities
├── models/                   # Network architecture definitions
│   └── GTACM.py              # Core GTACM model and frequency gating modules
├── physionet_data/           # EHR and ICU mortality dataset processing
│   ├── EHR Datasets Processing.ipynb
│   ├── Predicting Mortality of ICU Patients.ipynb
│   └── valid_subjects.pkl
├── timeseries/               # Dataset preprocessing notebooks
│   ├── Beijing Air Quality Data preprocessing.ipynb
│   └── Electricity Transformer Dataset (ETDataset) Data preprocessing.ipynb
├── utils/                    # Utility functions and loss definitions
│   ├── afall_loss.py         # Custom objective functions
│   └── missing_mechanisms.py # Synthetic missingness mask generation (MCAR, MAR, MNAR)
├── LICENSE                   # Project license
└── README.md                 # Project documentation

```

---

## 📊 Datasets & Preprocessing Pipelines

DyFAIP supports several benchmark datasets across healthcare, environmental, and industrial domains. Below is the detailed breakdown of the included data processing pipelines.

### 1. Beijing Air Quality Benchmark

* **Raw Data Location:** Stored in `AIR_QUALITY/` containing 12 monitoring station CSV files.


* **Raw Volume:** 420,768 records across 18 columns.


* **Filtered Subset:** Filtered to the temporal range `2013-03-01` to `2014-01-01`, producing **88,416 records**.


* **Target Variable:** `PM2.5` (Fine particulate matter concentration).


* **Feature Set:**
* **Environmental Pollutants:** `PM10`, `SO2`, `NO2`, `CO`, `O3`

* **Meteorological Measurements:** `TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`

* **Engineered Wind Features:** `cos_wind_direction`, `sin_wind_direction`

* **Temporal Indicators:** `month`, `hour`, `week_day_numeric`




#### Preprocessing & Feature Engineering Steps

:

1. **Multi-Station Concatenation:** Merges all 12 station CSV files into a unified dataset.


2. **Datetime Parsing:** Formats `year`, `month`, `day`, and `hour` into a single datetime index and derives numerical weekday features.


3. **Trigonometric Encoding:** Converts discrete compass wind direction strings into continuous sine and cosine angles (`sin_wind_direction`, `cos_wind_direction`) to preserve cyclic spatial characteristics.


4. **Metadata Cleanup:** Drops non-predictive identifiers (`No`, `year`, `dateInt`, `date`) while retaining essential weather and air quality variables.



---

### 2. Electricity Transformer Dataset (ETDataset)

* **Location:** Stored in `datasets/ETT-small/`.


* **Processing:** Managed through `timeseries/Electricity Transformer Dataset (ETDataset) Data preprocessing.ipynb`.



---

### 3. PhysioNet ICU Benchmark

* **Location:** Managed inside `physionet_data/`.


* **Task:** Clinical electronic health record (EHR) processing and ICU patient mortality prediction.



---

## 🛠 Prerequisites & Installation

Ensure your environment satisfies the version requirements below to prevent tensor operation mismatches.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/DyFAIP.git
cd DyFAIP

```

### 2. Environment Setup

We recommend using **Conda** to manage dependencies:

```bash
conda create -n dyfaip python=3.10 -y
conda activate dyfaip

```

### 3. Install Dependencies

Install the core framework along with necessary data processing packages:

```bash
pip install torch==2.5.1 numpy==2.0.1 scipy pandas scikit-learn matplotlib jupyter tqdm glob2

```

> **Required Dependencies:** `torch`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `glob`.
> 
> 

---

## 🚀 Getting Started

Execution consists of dataset preprocessing followed by model training.

### Step 1: Preprocess Datasets

Execute the appropriate Jupyter notebook in `timeseries/` or `physionet_data/` depending on your target benchmark:

* **Beijing Air Quality:** Run `timeseries/Beijing Air Quality Data preprocessing.ipynb`.


* **ETDataset:** Run `timeseries/Electricity Transformer Dataset (ETDataset) Data preprocessing.ipynb`.


* **PhysioNet EHR Data:** Run `physionet_data/EHR Datasets Processing.ipynb`.



### Step 2: Training & Evaluation

To train DyFAIP and run evaluation routines for imputation and prediction metrics:

```bash
python helpers/Runner.py

```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
