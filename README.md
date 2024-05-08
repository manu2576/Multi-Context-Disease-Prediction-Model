# Multi-Context Disease Prediction Model

## Overview
This project aims to predict diseases based on symptoms provided by users. The model predicts not only the disease but also suggests medications, workout routines, diet plans, and precautions associated with the predicted disease. The model utilizes Support Vector Classifier (SVC), fuzzy string matching, label encoding, and train-test split techniques to achieve accurate predictions across diverse contexts.

## Features
- Disease prediction based on user-provided symptoms.
- Recommendations for medications, workout routines, diet plans, and precautions.
- Utilization of fuzzy string matching for robust symptom matching.
- SVC model with an accuracy of 1 on both training and testing datasets.
- Label encoding for converting categorical data into numerical format.
- Utilization of NumPy and Pandas for data manipulation and preprocessing.
- Train-test split for evaluating model performance and generalization.

## Usage
1. **Input Symptoms**: Users input their symptoms separated by commas.
2. **Prediction**: The model predicts the disease based on the input symptoms.
3. **Recommendations**: Recommendations for medications, workout routines, diet plans, and precautions are provided for the predicted disease.

## Technologies Used
- Python: Programming language used for model development.
- scikit-learn: Library used for building and training the SVC model.
- fuzzywuzzy: Library used for fuzzy string matching to enhance symptom matching accuracy.
- NumPy: Library used for numerical computations and array manipulation.
- Pandas: Library used for data manipulation and preprocessing.
- Label Encoding: Technique used for converting categorical data into numerical format.
- Train-test split: Technique used for evaluating model performance and generalization.

## Dataset
- The dataset used for training and testing the model includes symptom-disease mappings along with associated medications, workout routines, diet plans, and precautions.

## Model Training
- The SVC model was trained on the dataset to predict diseases based on symptoms.
- Cross-validation techniques were used to ensure robustness and generalization of the model.

## Evaluation
- The model achieved an accuracy of 1 on both the training and testing datasets, indicating excellent performance and generalization.
