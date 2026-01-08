# Galaxy Morphology & Merging Classification 🌌🔭

This repository contains two Machine Learning projects focused on the morphological classification of galaxies, using data and images from the **Galaxy Zoo** project (SDSS/Euclid mission data).

The goal is to automate the identification of galaxy shapes and merger events, which is crucial for understanding the evolution of the universe.

## 🚀 Projects Overview

### 1. Multi-class Morphological Classification
An end-to-end pipeline to categorize galaxies into three primary types: **Spiral, Elliptical, and Merging**.
* **Methodology:** Extensive data cleaning, normalization, and comparison between **SVM** and **KNN** models.
* **Key Challenge:** Handling large datasets (380k+ records) and optimizing feature selection from physical measurements.

### 2. Binary Merger Prediction
A specialized model focused on the high-precision detection of **Galaxy Mergers**.
* **Methodology:** Implementation of ensemble methods using **XGBoost** and **Random Forest**.
* **Metrics:** Evaluated using F1-score and ROC-AUC to ensure robust detection of rare merger events.

---

## 📊 Dataset Information

The data and images used in these projects were sourced from the **Galaxy Zoo** project.
* **Source:** [Galaxy Zoo Data - Table 12 & Euclid](https://data.galaxyzoo.org/#section-24)
* **Description:** The dataset includes morphological classifications and physical parameters derived from astronomical surveys.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib.
* **Environment:** Jupyter Notebooks for detailed step-by-step analysis.

---

## 📈 Key Results

* Successful distinction between Spiral and Elliptical morphologies with high accuracy.
* Improved merger detection using XGBoost feature importance analysis.
* Visualization pipeline for direct image inspection from compressed astronomical data.

---

## 🔮 Future Work: The Next Step

The project is evolving. The next phase will transition from tabular data to **Deep Learning**. I am currently working on:
* Training a **Convolutional Neural Network (CNN)** to perform image-based classification.
* Implementing Computer Vision techniques to automate feature extraction directly from SDSS/Euclid images.

---

## 📬 Contact & Links

* **GitHub:** [MiguelPelaezR](https://github.com/MiguelPelaezR)
* **Project Repository:** [clasificacion_galaxias](https://github.com/MiguelPelaezR/clasificacion_galaxias)
* **LinkedIn:** [Miguel Pelaez](https://www.linkedin.com/in/miguel-pelaez-62b8aa268/)

---
*Developed as part of my Data Science Portfolio.*
