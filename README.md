# 🌊 Andaman & Nicobar Islands Earthquake Risk Forecast

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning-based earthquake risk forecasting system for the Andaman and Nicobar Islands region using historical seismic data from USGS.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Data](#data)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🔍 Overview

This project implements a time-series based classification model to forecast earthquake risk in the Andaman and Nicobar Islands region. The system uses historical earthquake data to predict the likelihood of significant seismic events, helping to improve early warning systems and disaster preparedness.

**Key Highlights:**
- Uses Histogram-based Gradient Boosting Classifier
- Time-based feature engineering from seismic event patterns
- Temporal validation split to avoid data leakage
- ROC-AUC and Brier score evaluation metrics
- Production-ready modular code structure

## ✨ Features

- **Data Processing Pipeline**: Automated ETL for USGS earthquake data
- **Feature Engineering**: Time-based aggregations and rolling statistics
- **Machine Learning Model**: HistGradientBoostingClassifier with calibrated probabilities
- **Model Evaluation**: Comprehensive metrics including calibration curves
- **Modular Design**: Clean separation of concerns for easy maintenance
- **Reproducible**: Fixed random seeds and version-controlled dependencies

## 📁 Project Structure

```
andaman-nicobar-quake-risk-forecast/
│
├── data/
│   ├── raw/                    # Raw earthquake data from USGS
│   ├── processed/              # Processed parquet files
│   └── models/                 # Trained model artifacts
│
├── notebooks/
│   └── 01_report.ipynb         # Analysis and reporting notebook
│
├── src/eqf/                    # Source code package
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── make_dataset.py         # Data loading and preprocessing
│   ├── features.py             # Feature definitions
│   ├── build_features.py       # Feature engineering pipeline
│   ├── train.py                # Model training script
│   └── predict.py              # Inference script
│
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project metadata and build config
└── README.md                  # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/junaidreal4-hub/andaman-nicobar-quake-risk-forecast.git
cd andaman-nicobar-quake-risk-forecast
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install the package in development mode**
```bash
pip install -e .
```

## 💻 Usage

### 1. Data Preparation

Process the raw earthquake data:

```bash
python -m eqf.make_dataset
```

This script:
- Loads raw USGS earthquake data
- Validates required columns
- Converts timestamps
- Filters earthquake events
- Saves processed data as Parquet

### 2. Feature Engineering

Generate features from processed data:

```bash
python -m eqf.build_features
```

Creates time-based features including:
- Event counts over different time windows (1d, 7d, 30d)
- Maximum magnitudes in time windows
- Mean depth statistics

### 3. Model Training

Train the forecasting model:

```bash
python -m eqf.train
```

Outputs:
- Trained model saved to `data/models/`
- Calibration table in `data/processed/`
- Test predictions for evaluation
- Performance metrics (ROC-AUC, Brier score)

### 4. Making Predictions

Generate forecasts for new data:

```bash
python -m eqf.predict
```

## 🔬 Methodology

### Data Source
Historical earthquake data from the **United States Geological Survey (USGS)** covering the Andaman and Nicobar Islands region.

### Feature Engineering
The model uses temporal aggregation features:
- **count_1d, count_7d, count_30d**: Number of seismic events in the past 1, 7, and 30 days
- **maxmag_1d, maxmag_7d, maxmag_30d**: Maximum magnitude recorded in time windows
- **mean_depth**: Average depth of recent earthquakes

### Model Architecture
- **Algorithm**: HistGradientBoostingClassifier
- **Hyperparameters**: 
  - max_depth: 6
  - learning_rate: 0.05
- **Validation Strategy**: Time-based split (no shuffling to prevent data leakage)
- **Calibration**: Probability calibration using calibration curves (10 bins, uniform strategy)

### Evaluation Metrics
- **ROC-AUC Score**: Measures the model's ability to distinguish between classes
- **Brier Score Loss**: Evaluates probability calibration quality
- **Classification Report**: Precision, recall, and F1-score
- **Calibration Curve**: Visual assessment of probability reliability

## 📊 Data

### Data Schema

Required columns in raw data:
- `time`: Timestamp of the earthquake event
- `latitude`: Geographic latitude
- `longitude`: Geographic longitude
- `depth`: Depth of the earthquake (km)
- `mag`: Magnitude of the earthquake
- `type`: Event type (filtered for 'earthquake')

### Data Acquisition

Data can be downloaded from [USGS Earthquake Catalog](https://earthquake.usgs.gov/earthquakes/search/).

## 📈 Model Performance

*Performance metrics will be displayed here after training. The model is evaluated using:*

- **ROC-AUC Score**: Measures discrimination ability
- **Brier Score**: Measures calibration quality (lower is better)
- **Calibration Plot**: Shows reliability of predicted probabilities

*Example output from training:*
```
Test ROC-AUC: 0.85
Brier score loss: 0.12
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to functions and classes
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Mohammed Junaid Khan**
- GitHub: [@junaidreal4-hub](https://github.com/junaidreal4-hub)
- Project Link: [https://github.com/junaidreal4-hub/andaman-nicobar-quake-risk-forecast](https://github.com/junaidreal4-hub/andaman-nicobar-quake-risk-forecast)

## 🙏 Acknowledgments

- **USGS** for providing comprehensive earthquake data
- **scikit-learn** community for excellent ML tools
- Inspired by seismology research and early warning systems

---

⭐ If you find this project useful, please consider giving it a star!

**Disclaimer**: This model is for research and educational purposes. It should not be used as the sole basis for critical safety decisions. Always consult official seismological agencies for earthquake warnings and safety information.
