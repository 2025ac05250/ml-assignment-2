# Machine Learning Assignment 2

**Student ID:** 2025AC05250  
**Name:** Vasu Gupta

## 1. Problem Statement

Build and evaluate multiple classification models on a public dataset and use the trained models to predict whether an individual's annual income exceeds $50K.

## 2. Dataset Description

**Dataset:** UCI Adult Income  
**Problem type:** Binary classification  
**Target:** `income`

The dataset contains demographic and employment-related information. After removing rows with missing values, the dataset used for this assignment contains **30,162 instances** and **14 input features**.

## 3. GitHub Repository Link

**Repository:** https://github.com/2025ac05250/ml-assignment-2.git

The repository contains the notebook, Streamlit application, requirements file, test data, and saved model files.

## 4. Streamlit App Link

**URL:** https://2025ac05250-ml-assignment-2.streamlit.app/

## 5. Models and Evaluation

### Models Used

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbor (kNN)
- Gaussian Naive Bayes
- Random Forest (Ensemble)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8475 | 0.9022 | 0.7354 | 0.6052 | 0.6640 | 0.5711 |
| Decision Tree | 0.8462 | 0.8835 | 0.7315 | 0.6039 | 0.6616 | 0.5675 |
| kNN | 0.8270 | 0.8595 | 0.6664 | 0.6105 | 0.6372 | 0.5248 |
| Naive Bayes | 0.7873 | 0.8280 | 0.6617 | 0.2983 | 0.4112 | 0.3394 |
| Random Forest (Ensemble) | 0.8492 | 0.9012 | 0.7280 | 0.6292 | 0.6750 | 0.5801 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Gives the highest AUC (0.9022) and precision (0.7354). Its F1-score of 0.6640 is slightly lower than Random Forest. |
| Decision Tree | Performs close to Logistic Regression in accuracy and F1-score, but has a lower AUC of 0.8835. |
| kNN | Gives reasonable precision and recall, but its accuracy (0.8270), AUC (0.8595) and F1-score (0.6372) are lower than the other models except Naive Bayes. |
| Naive Bayes | Has the lowest overall performance, particularly in recall (0.2983) and F1-score (0.4112). |
| Random Forest (Ensemble) | Achieves the highest accuracy (0.8492), recall (0.6292), F1-score (0.6750) and MCC (0.5801), giving the best overall performance based on F1-score. |

### Overall Winner

**Random Forest (Ensemble)** is the overall winner based on the highest F1-score of **0.6750** on the held-out test set.
