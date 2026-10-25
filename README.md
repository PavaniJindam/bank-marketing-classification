# Bank Marketing: Comparative Analysis of Classification Algorithms

## Project Overview
This Fundamentals of Data Science case study analyzes bank marketing data to predict whether a customer will subscribe to a term deposit.

## Objectives
- Understand and preprocess the dataset.
- Explore customer and campaign characteristics.
- Train and compare four classification algorithms.
- Evaluate models using accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC.
- Recommend a model based on the evaluation results.

## Dataset
The dataset contains 4,521 customer records and 17 columns. The target variable is `y`, which indicates whether a customer subscribed to a term deposit (`yes` or `no`). The CSV file uses a semicolon delimiter.

## Algorithms
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbours (KNN)
4. Random Forest Classifier

The `duration` feature is excluded from the primary prediction models because it is not known before a marketing call takes place.

## Evaluation
The models are compared using accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC. Model selection considers the business impact of identifying potential subscribers and avoiding unnecessary marketing calls.

## Files
- `Bank_Marketing_Classification.ipynb` — notebook containing analysis, code, outputs, and visualizations.
- Python script (`.py`) — source code.
- `requirements.txt` — required Python libraries.

## Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

## How to Run
1. Install the required libraries listed in `requirements.txt`.
2. Download the dataset and place it in the location expected by the notebook or update the file path.
3. Open the notebook in Jupyter Notebook or Google Colab.
4. Run the cells in order.
