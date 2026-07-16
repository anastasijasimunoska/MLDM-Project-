# Analysis of Customer Shopping Behaviour Using Data Mining Methods

Course project for Machine Learning and Data Mining, analyzing customer shopping behaviour using three complementary data mining techniques, following the CRISP-DM methodology.

**Author:** Anastasija Simunoska

## Overview

This project applies:
- **Clustering (K-means)** — identifies natural customer segments based on shopping behaviour
- **Classification (Artificial Neural Network + Rule-Based classifier)** — predicts a customer's Subscription Status
- **Association Analysis (Apriori)** — discovers relationships between purchased product category, season, and size

Clustering is performed first, so the resulting customer segments can be used as an additional input feature for classification.

## Dataset

[Shopping Behaviour Dataset](https://www.kaggle.com/datasets/saadaliyaseen/shopping-behaviour-dataset) (Kaggle, saadaliyaseen) — 3,900 customer records across 18 attributes.

## Key Findings

- **Gender, Discount Applied, and Promo Code Used** were found to be near-perfect predictors of Subscription Status and were deliberately excluded from the classification models to keep the task meaningful.
- **K-means (k=4)** identified four customer segments, distinguished primarily by age and spending level rather than product preferences.
- **Classification results:** the ANN achieved higher overall accuracy, while the rule-based classifier achieved a better F1 score by catching more real subscribers (higher recall), illustrating a classic precision-recall tradeoff.
- **Association analysis** found only weak associations (lift 0.92–1.09) between category, season, and size, indicating these attributes are largely independent in this dataset.

## Tools

Python (Kaggle Notebook) — pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow.keras. The Apriori algorithm was implemented by hand (no external libraries), since installing packages required internet access unavailable in the Kaggle session.

## Run it

**Notebook:** `mldm-project.ipynb` (in this repo)
