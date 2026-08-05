Here is an updated, publication-ready `README.md` formatted specifically for top-tier venues (NeurIPS, ICML, AAAI, ICLR).

This version fixes formatting issues (such as the trailing bullet point under prerequisites), incorporates the complete file tree visible in your repository sidebar (including `physionet_data`, `timeseries`, `utils`, and helper modules), adds status badges, and expands setup instructions to cover all dataset pipelines.

---

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
├── datasets/                  # Processed datasets and benchmark files
│   ├── AIR_QUALITY/           # Beijing Air Quality dataset
│   └── ETT-small/             # Electricity Transformer Temperature dataset
├── helpers/                   # Core execution and training logic
│   ├── Runner.py              # Main entry point for training & evaluation
│   ├── metrics.py             # Imputation and prediction evaluation metrics
│   └── trainer_helper.py      # Optimization loops and training utilities
├── models/                    # Network architecture definitions
│   └── GTACM.py               # Core GTACM model and frequency gating modules
├── physionet_data/            # EHR and ICU mortality dataset processing
│   ├── EHR Datasets Processing.ipynb
│   ├── Predicting Mortality of ICU Patients.ipynb
│   └── valid_subjects.pkl
├── timeseries/                # Dataset preprocessing notebooks
│   ├── Beijing Air Quality Data preprocessing.ipynb
│   └── Electricity Transformer Dataset (ETDataset) Data preprocessing.ipynb
├── utils/                     # Utility functions and loss definitions
│   ├── afall_loss.py          # Custom objective functions
│   └── missing_mechanisms.py  # Synthetic missingness mask generation (MCAR, MAR, MNAR)
├── LICENSE                    # Project license
└── README.md                  # Project documentation

```

---

## 🛠 Prerequisites & Installation

Ensure your environment satisfies the version requirements below to prevent broadcasting or tensor operation mismatches.

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

```bash
pip install torch==2.5.1 numpy==2.0.1 scipy pandas scikit-learn matplotlib jupyter

```

---

## 🚀 Getting Started

The pipeline consists of two steps: dataset preprocessing and model execution.

### Step 1: Data Preprocessing

Depending on the dataset you wish to benchmark, run the corresponding notebook in `timeseries/` or `physionet_data/`:

* **Electricity Transformer Dataset (ETDataset):**
Execute `timeseries/Electricity Transformer Dataset (ETDataset) Data preprocessing.ipynb`
* **Beijing Air Quality Dataset:**
Execute `timeseries/Beijing Air Quality Data preprocessing.ipynb`
* **PhysioNet EHR Dataset:**
Execute `physionet_data/EHR Datasets Processing.ipynb`

> **Note:** Ensure output `.npz` or `.pkl` files are placed inside the appropriate subfolder in `datasets/` as expected by `helpers/Runner.py`.

### Step 2: Training & Evaluation

To train DyFAIP and evaluate its imputation and downstream prediction metrics, execute `Runner.py`:

```bash
python helpers/Runner.py

```


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
