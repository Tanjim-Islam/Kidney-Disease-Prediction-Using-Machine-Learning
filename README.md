# Kidney Disease Prediction Using Logistic Regression, Decision Trees, Random Forest, SVM, XGBoost, and Gradient Boosting

## Project Overview

This project explores how effectively different machine learning models can predict the likelihood of chronic kidney disease (CKD) based on health indicators. The objective is to experiment with multiple models and assess their accuracy. The project involves data preprocessing, feature engineering, and applying models such as Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, and Gradient Boosting.

## Features

- **Binary Classification**: Predict whether the patient has CKD (0) or does not have CKD (1).
- **Data Visualizations**: Includes histograms, count plots, and violin plots for insights into the dataset.
- **Preprocessing**: Handles missing values and encodes categorical variables.
- **Model Evaluation**: Uses accuracy, confusion matrices, classification reports, and ROC curves to assess model performance.
- **Comparison of Multiple Models**: Evaluates models with hyperparameter tuning.

## Dataset

- **Kidney Disease Dataset** (`kidney_disease.csv`): Contains various health features related to kidney disease prediction.

## Requirements

- **Python 3.10**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Matplotlib**
- **Seaborn**
- **Plotly**

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/Tanjim-Islam/Kidney-Disease-Prediction-Using-Machine-Learning.git
    cd Kidney-Disease-Prediction-Using-Machine-Learning
    ```

2. **Install Dependencies:**

    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly
    ```

3. **Ensure the Dataset is Available:**
   Place `kidney_disease.csv` in the working directory.

## Code Structure

1. **Data Preprocessing:**
   - Handle missing values with random sampling or mode imputation.
   - Encode categorical features using `LabelEncoder`.

2. **Exploratory Data Analysis (EDA):**
   - Use histograms, count plots, violin plots, and correlation heatmaps.

3. **Model Building and Evaluation:**
   - Train and evaluate the following models:
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
     - Decision Tree
     - Random Forest
     - XGBoost
     - Gradient Boosting

4. **Hyperparameter Tuning:**
   - Perform GridSearchCV to optimize Decision Tree, SVM, and Gradient Boosting models.

5. **Performance Metrics:**
   - **Accuracy**: Measure how well each model performs.
   - **Confusion Matrix**: Show correct and incorrect predictions.
   - **Classification Report**: Provide precision, recall, and F1-score.
   - **ROC Curves**: Visualize the trade-off between sensitivity and specificity.

## Results

### Model Accuracy Comparison:

| **Model**              | **Training Accuracy (%)** | **Testing Accuracy (%)** | **ROC AUC (%)** |
|------------------------|---------------------------|--------------------------|----------------|
| Logistic Regression    | 89.69                     | 91.25                    | 90.80          |
| KNN                    | 78.13                     | 63.75                    | 63.05          |
| SVM                    | 98.75                     | 73.75                    | 74.04          |
| Decision Tree (DT)     | 98.44                     | 97.50                    | 96.29          |
| Random Forest          | 99.38                     | 97.50                    | 96.43          |
| XGBoost                | 98.13                     | 97.50                    | 96.43          |
| Gradient Boosting      | 100.00                    | 97.50                    | 96.43          |

### Performance Evaluation Plot:
A bar plot comparing the accuracy and ROC-AUC scores of various models:

![Performance Evaluation - Kidney Disease Prediction](Images/PE_kidney.jpeg)

### ROC Curves:
ROC curves showing the sensitivity vs. specificity for each model:

![ROC - Kidney Disease Prediction](Images/roc_kidney.jpeg)
