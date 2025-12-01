


<div align="center">

![TravelTide Logo](reports/traveltide_logo.png)

# TravelTide: Customer Segmentation & Personalization Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Status](https://img.shields.io/badge/status-production-green.svg)

> **A production-ready, modular framework for intelligent customer segmentation, perk assignment, and A/B testing—built for real-world travel analytics.**

[![View Presentation](https://img.shields.io/badge/📊-View_Presentation-blue?style=for-the-badge)](reports/docs/presentation_traveltide_rewards.pdf)
[![Read Full Report](https://img.shields.io/badge/📄-Read_Full_Report-green?style=for-the-badge)](reports/docs/raports_summary.pdf)

![TravelTide Analytics](reports/traveltide_copy.png)

</div>
> **A production-ready, modular framework for intelligent customer segmentation, perk assignment, and A/B testing—built for real-world travel analytics.**

---

## 📋 Table of Contents

* [Overview](#overview)
* [TravelTide Datasets](#traveltide-datasets)
* [Elena’s Cohort Definition](#elenas-cohort-definition)
* [Key Features](#key-features)
* [Strategic Insights & Findings](#strategic-insights--findings)
* [Project Architecture](#project-architecture)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Data Workflow](#data-workflow)
* [Segmentation Approaches](#segmentation-approaches)
* [A/B Testing Framework](#ab-testing-framework)
* [Configuration](#configuration)
* [API Reference](#api-reference)
* [Examples](#examples)
* [Testing](#testing)
* [Performance](#performance)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Project Status](#project-status)

---

## 🎯 Overview

TravelTide is a **unified customer intelligence platform** designed for the travel industry.
It combines **rule-based segmentation**, **unsupervised machine learning**, and a **statistical A/B testing engine** to deliver personalized perks, boost conversions, and optimize marketing spend.

### What TravelTide Delivers

* Actionable segmentation based on **demographics, behavior, and travel patterns**
* Identification of **high-value user groups** (e.g., VIP High-Frequency Spenders)
* Automated **perk assignment** across campaigns
* Robust **comparison tools** (ARI, NMI, V-Measure, Fowlkes–Mallows)
* A/B testing of perk strategies to quantify **subscription lift**
* End-to-end **workflow from data extraction → processing → segmentation → analysis**

---

## 🧬 TravelTide Datasets

The platform integrates multiple raw and processed datasets from TravelTide’s PostgreSQL environment.

### **Raw Tables (Source Layer)**

* **`users`** – User demographics
* **`sessions`** – Browsing & platform interaction data
* **`flights`** – All flight bookings
* **`hotels`** – All hotel reservations

### **Processed Tables (Analytics Foundation)**

* **`sessions_cleaned`**

  * Cleaned & normalized browsing sessions
  * **49,211 sessions** after processing
* **`sessions_not_canceled_trips`**

  * Filtered to non-canceled trips
  * **14,895 valid sessions** (model-ready)
* **Feature Metrics** (final feature store):

  * `num_clicks`
  * `avg_session_duration`
  * `conversion_rate`
  * `RFM_score`
  * `persona_type`

These curated tables feed into segmentation and A/B testing.

---

## 👤 Elena’s Cohort Definition

Elena’s logic defines the **high-intent cohort** used for deeper modeling and experimentation.

**Inclusion Criteria:**

* Sessions on or after **January 4, 2023**
* Users with **>7 sessions**
* Enriched with flight & hotel booking details

**Cohort Summary:**

* **5,998 unique users**
* **49,211 sessions** generated
* **16,099 total trips**
* Represents TravelTide’s **most engaged, highest-conversion audience**

This provides a reliable statistical base for ML clustering and perk optimization.

---

## ✨ Key Features

### 🔀 Dual Segmentation Engine

| Feature      | Rule-Based                 | Machine Learning             |
| ------------ | -------------------------- | ---------------------------- |
| Transparency | ⭐⭐⭐⭐⭐                      | ⭐⭐⭐                          |
| Scalability  | ⭐⭐⭐                        | ⭐⭐⭐⭐⭐                        |
| Flexibility  | ⭐⭐⭐                        | ⭐⭐⭐⭐⭐                        |
| Best Use     | Compliance, manual control | Behavioral pattern detection |

### 🔧 Core Capabilities

* **50+ engineered features** (behavioral + transactional)
* **K-Means and DBSCAN clustering pipelines**
* **Consistency metrics** (ARI, NMI, V-Measure)
* **Perk Recommendation Engine**
* **End-to-end A/B testing**
* **High-quality plots and reporting utilities**
* Export to **CSV, JSON, and HTML dashboards**

---

## 💡 Strategic Insights & Findings

Based on `analysez.ipynb`, `comparison.ipynb`, and `perk_ab_test.ipynb`:

### 1️⃣ ML Clusters Are Behaviorally Stronger

Machine learning segmentation uncovers **cleaner, more cohesive behavioral segments**, outperforming demographic-only manual groups.

### 2️⃣ Manual Segments Need Refinement

Some rule-based personas (e.g., *Family*, *Couple*) show inconsistent behavioral patterns and low operational value due to small size.

### 3️⃣ Hybrid Approach Wins

Use:

* **Manual segments** for *messaging & communication*
* **ML clusters** for *targeting & perk eligibility*

This maximizes interpretability *and* performance.

### 4️⃣ Highest-Value Segment Confirmed

The **VIP High-Frequency Spenders** segment has an **average spend of $8,371.94**, making it ideal for premium retention perks.

---

## 🏗 Project Architecture

```
traveltide_project/
│
├── 📁 config/                      # Configuration files
│   ├── ml_config.yaml              # ML model parameters
│   └── non_ml_config.yaml          # Rule-based thresholds
│
├── 📁 data/                        # Data storage
│   ├── csv/
│   │   ├── raw/                    # Original datasets
│   │   │   ├── elena_cohort.csv
│   │   │   ├── flights.csv
│   │   │   ├── hotels.csv
│   │   │   ├── sessions.csv
│   │   │   └── users.csv
│   │   └── processed/              # Cleaned & engineered data
│   │       ├── feature/
│   │       └── segment/
│   └── sql/                        # SQL extraction scripts
│
├── 📁 src/                         # Source code
│   ├── db.py                       # Database utilities
│   ├── utils.py                    # Helper functions
│   │
│   └── core/                       # Core modules
│       ├── features/               # Feature engineering
│       │   ├── user_behavior_metrics.py
│       │   ├── user_advanced_metrics.py
│       │   └── user_feature_pipeline.py
│       │
│       ├── processing/             # Data processing
│       │   ├── load_data.py
│       │   ├── session_cleaner.py
│       │   └── eda.py
│       │
│       └── segment/                # Segmentation engines
│           ├── ml_model/           # ML-based segmentation
│           │   ├── clustering_orchestrator.py
│           │   ├── kmeans_engine.py
│           │   ├── dbscan_engine.py
│           │   ├── feature_engineer.py
│           │   ├── perk_assigner.py
│           │   ├── metrics_calculator.py
│           │   └── visualizer.py
│           │
│           ├── non_ml/             # Rule-based segmentation
│           │   ├── non_machine_learning_segment.py
│           │   ├── threshold_manager.py
│           │   ├── perk_assigner.py
│           │   └── analyzer.py
│           │
│           ├── comparison/         # Compare approaches
│           │   └── segmentation_comparator.py
│           │
│           └── ab_test/            # A/B testing framework
│               ├── ab_test_framework.py
│               └── statistical_tests.py
│
├── 📁 notebooks/                   # Analysis notebooks
│   ├── features/
│   ├── preparing_data/
│   └── segments/
│
├── 📁 reports/                     # Generated outputs
│   └── segment/
│       ├── ml_model/
│       ├── non_ml/
│       ├── comparison/
│       └── ab_test/
│
├── setup.py                        # Package setup
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## ⚡ Quick Start

### 1. ML Segmentation

```python
from src.core.segment.ml_model import MLClustering
import pandas as pd

df = pd.read_csv("data/csv/processed/user_base.csv")
ml = MLClustering(config_path="config/ml_config.yaml")
results = ml.run_kmeans(df, n_clusters=5)
```

### 2. Compare Manual vs ML Segments

```python
from src.core.segment.comparison import SegmentationComparator

manual = pd.read_csv(".../non_ml/customer_segmentation_detailed.csv")
ml = pd.read_csv(".../ml_model/kmeans_segmentation.csv")

comp = SegmentationComparator(manual, ml)
analysis = comp.run_complete_analysis()

print(analysis["metrics"])
print(analysis["recommendations"][0])
```

### 3. A/B Test Strategies

```python
from src.core.segment.ab_test import ABTestFramework

ab = ABTestFramework(manual, ml)
groups = ab.create_test_groups()
results = ab.analyze_results()

print(results["recommendation"])
```

---

## 🧪 A/B Testing Framework

Supports:

* χ² tests
* Fisher exact tests
* t-tests
* Mann–Whitney U tests

**Test Groups:**

* **A** – Manual perk assignment
* **B** – ML-based perk assignment
* **C** – Randomized control

The framework recommends a winner based on statistical significance and business KPIs.

---

## 🔧 Configuration

Two configuration layers:

* `ml_config.yaml` – Clustering parameters
* `non_ml_config.yaml` – Business-rule thresholds

Supports full plug-and-play customization.

---

## 📚 API Reference

Full API documentation is available via docstrings and the `src/` module organization.

---

## 🧪 Testing

The system includes:

* Unit tests
* Integration tests
* Synthetic simulation testing for perk distribution and segment consistency

---

## 📈 Performance

Designed for:

* **100k+ users**
* Fast clustering
* Modular incremental retraining

---

## 🤝 Contributing

Pull requests welcome.
Make sure code is formatted with **Black** and passes all tests.

---

## 📄 License

MIT License.

---

## 📬 Contact

For questions, issues, or collaboration:

* GitHub Issues
* Email (if applicable)

---

## 📊 Project Status

* **Version:** 1.0.0
* **Status:** Production Ready
* **Last Updated:** November 2025

### Roadmap

* [x] Rule-based segmentation
* [x] ML clustering (K-Means, DBSCAN)
* [x] Comparison framework
* [x] A/B test engine
* [ ] Real-time segmentation API
* [ ] Supervised uplift modeling
* [ ] Deep learning embeddings
* [ ] Automated retraining pipeline
* [ ] ML → Manual **Segment Consolidation System** (from comparison findings)


