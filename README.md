# Iranian Protest Narrative

This repository contains the code used for the chapter  
**“Narrating Protest for Visibility: The Iranian Uprising on Twitter”**  
from the doctoral thesis of **Melody Sepahpour-Fard**.

The analyses examine how narratives surrounding the Iranian protests — following the death of Jina Mahsa Amini — emerged and evolved on Twitter. The scripts and notebooks included here were used to identify duplicate content, explore thematic structures, and visualize temporal and topical trends in both Persian- and English-language Twitter datasets.

---

## 📘 Overview

The notebooks and scripts in this repository perform the following main steps:

1. **Duplicate detection**  
   Identifies and removes near-duplicate tweets in both Persian and English datasets.

2. **Topic modeling and narrative extraction**  
   Uses BERTopic to identify and interpret recurring themes and narratives.

3. **Exploratory data analysis (EDA)**  
   Investigates the distribution, frequency, and content of tweets over time.

4. **Robustness checks and threshold optimization**  
   Tests and validates thresholds for duplicate detection.

5. **Visualization and graph generation**  
   Creates network graphs and trend figures to represent relationships between narratives and their visibility dynamics.

---

## 🧩 Files and Their Purpose

| File | Description |
|------|--------------|
| **`duplicate_detection_persian.py`** | Detects and filters duplicate Persian-language tweets. |
| **`duplicate_detection_english.py`** | Detects and filters duplicate English-language tweets. |
| **`eda_duplicates_fulldata.ipynb`** | Exploratory Data Analysis of duplicate tweets. |
| **`find_best_duplicate_threshold_english.ipynb`** | Tests different similarity thresholds for English duplicate detection. |
| **`robustness_check_duplicate_threshold_persian.ipynb`** | Validates threshold robustness for Persian duplicate detection. |
| **`bertopic_english_duplicate.ipynb`** | Performs BERTopic modeling on English tweets. |
| **`create_graph_largest_duplicate.ipynb`** | Builds and visualizes the largest connected component of the narrative graph. |
| **`create_labelling_sample_english.ipynb`** | Prepares a sample of English tweets for manual labeling. |
| **`google_trends_figure.ipynb`** | Generates a comparative figure linking protests in Iran with Google Trends data. |

---

## ⚙️ Dependencies

These analyses require Python 3.8+ and the following libraries:
pandas
numpy
scikit-learn
matplotlib
seaborn
tqdm
sentence-transformers
bertopic
umap-learn
hdbscan
plotly

🗂️ Repository Structure
iranian_protest_narrative/
│
├── bertopic_english_duplicate.ipynb
├── create_graph_largest_duplicate.ipynb
├── create_labelling_sample_english.ipynb
├── duplicate_detection_english.py
├── duplicate_detection_persian.py
├── eda_duplicates_fulldata.ipynb
├── find_best_duplicate_threshold_english.ipynb
├── google_trends_figure.ipynb
├── robustness_check_duplicate_threshold_persian.ipynb
└── README.md

✳️ Author: Melody Sepahpour-Fard
📍 Project: Thesis chapter – Narrating Protest for Visibility: The Iranian Uprising on Twitter
📅 Year: 2025

