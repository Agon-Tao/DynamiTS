# 🌌 DynamiTS 

### DynamiTS: A Structure-Guided Framework for Multivariate Time Series Forecasting via Adaptive Multi-Scale Fusion and Dynamic Patch Expansion

<p align="center">
  <b>Adaptive Scale Selection · Long-term Forecasting · Mixture-of-Experts · Robust Temporal Modeling</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-1.10+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/Task-Time%20Series%20Forecasting-00A67E?style=for-the-badge">
  <img src="https://img.shields.io/badge/License-MIT-FCC624?style=for-the-badge">
</p>
<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#main-results">Results</a> •
  <a href="#dataset-links">Dataset Links</a> •
  <a href="#usage">Usage</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#contact">Contact</a>
</p>

## 🔥 Highlights
<table>
<tr>
<td width="33%" align="center">

### 🧠 Adaptive Multi-Scale Fusion
Adaptively aggregate multi-resolution temporal representations.

</td>
<td width="33%" align="center">

### ⚡ Structure-Guided Dynamic Patch Expansion

Preserve local structural continuity through adaptive patch boundary adjustment

</td>
<td width="33%" align="center">

### 📈 Temporal-Aware DLinear Mixture of Experts

Capture variable-specific temporal dynamics with expert routing

</td>
</tr>
</table>


## 📌 Introduction

This repository provides the official implementation of **DynamiTS**, a deep learning model for long-term time series forecasting.


<h2 id="overview" name="overview">✨ Overview</h2>

**DynamiTS** is a deep learning framework designed for **long-term time series forecasting**.  
It introduces an adaptive multi-scale modeling strategy to capture both short-term fluctuations and long-term temporal dependencies.

Our model is built on three key ideas:

- **Adaptive Multi-Scale Fusion** for dynamically aggregating multi-resolution temporal representations.
- **Structure-Guided Dynamic Patch Expansion** for preserving local structural continuity through adaptive patch boundary adjustment.
- **Temporal-Aware DLinear Mixture of Experts** for capturing variable-specific temporal dynamics with expert routing.

<p align="center">
  <img src="./assets/structal_intensity_01.png" width="900" alt="Model Architecture">
</p>

<p align="center">
  <img src="./assets/dynamic patch_01.png" width="900" alt="Model Architecture">
</p>



<h2 id="architecture" name="architecture">🏗 Architecture</h2>

<p align="center">
  <img src="./assets/Model.png" width="900" alt="Model Architecture">
</p>


<h2 id="main-results" name="main-results">📊 Main Results</h2>

### Main Forecasting Results
<p align="center">
  <img src="./assets/results.png" width="850">
</p>

---


## 🧪 Selector Weight Analysis

<p align="center">
  <img src="./assets/Weather_512_to_192.png" width="850">
</p>

The learned selector weights reveal how the model adaptively focuses on different temporal scales for each input sample.

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```
---

<h2 id="dataset-links" name="dataset-links">📦 Dataset Links</h2>

The datasets used in this project can be downloaded from the following sources:

| Dataset | Source |
|---|---|
| ETT  | [ETDataset](https://github.com/zhouhaoyi/ETDataset) |
| Weather | [Weather Long-term Time Series Forecasting](https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting) |
| Traffic | [Traffic Hourly Dataset](https://hf-mirror.com/datasets/LeoTungAnh/traffic_hourly) |
| Exchange Rate | [Multivariate Time Series Data](https://github.com/laiguokun/multivariate-time-series-data) |
| Electricity | [UCI ECL Load Diagrams 2011-2014](https://archive.ics.uci.edu/ml/datasets/ECLLoadDiagrams20112014) |
| PEMS | [Caltrans Performance Measurement System](https://pems.dot.ca.gov) |
| METR-LA | [METR-LA Dataset](https://github.com/liyaguang/DCRNN) |
| Power | [Wind Solar Electricity Production](https://www.kaggle.com/datasets/henriupton/wind-solar-electricity-production) |

Please place the downloaded datasets under the `./data/` directory and make sure the file paths are consistent with the provided scripts.

Example data structure:

```text
data/
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
├── ETTm2.csv
├── weather.csv
├── electricity.csv
├── exchange_rate.csv
└── traffic/
    └── traffic.csv
```

<h2 id="usage" name="usage">🚀 Usage</h2>

### Train and evaluate on Dataset
```bash
bash ./scripts/ETTh1/ETTh1.sh
bash ./scripts/ETTh2/ETTh2.sh
bash ./scripts/ETTm1/ETTm1.sh
bash ./scripts/ETTm2/ETTm2.sh
bash ./scripts/ECL/ECL.sh
bash ./scripts/Traffic/Traffic.sh
bash ./scripts/Weather/Weather.sh
```


<h2 id="project-structure" name="project-structure">📁 Project Structure</h2>

```text
DynamiTS-main/
├── README.md
├── requirements.txt
├── main.py
├── LICENSE
├── .gitignore
│
├── assets/
│   └── figures and visualization images
│
├── checkpoints/
│   └── saved model checkpoints
│
├── models/
│   ├── __init__.py
│   ├── Amsf.py
│   ├── Common.py
│   ├── Dmoe.py
│   ├── DynamiTS.py
│   └── Uncertainty.py
│
├── scripts/
│   ├── ECL/
│   │   └── ECL.sh
│   ├── ETTh1/
│   │   └── ETTh1.sh
│   ├── ETTh2/
│   │   └── ETTh2.sh
│   ├── ETTm1/
│   │   └── ETTm1.sh
│   ├── ETTm2/
│   │   └── ETTm2.sh
│   ├── Exchange/
│   │   └── Exchange.sh
│   ├── Traffic/
│   │   └── Traffic.sh
│   └── Weather/
│       └── Weather.sh
│
└── utils/
    ├── __init__.py
    ├── dataloader.py
    ├── general.py
    └── tools.py
```
---


## 🙏 Acknowledgement

This project is inspired by the following excellent repositories:

- iTransformer
- PatchTST
- TimesNet
- Autoformer
- Informer
---

<h2 id="contact" name="contact">📬 Contact</h2>

For questions or suggestions, please contact:
```text
sunyujuan@ldu.edu.cn
```

---

<div align="center">

### ⭐ Star this repository if you find it useful.

</div>
