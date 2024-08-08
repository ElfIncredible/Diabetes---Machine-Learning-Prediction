# Diabetes - Machine Learning Prediction

The objective of the dataset:
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

The Pima Indian Diabetes data set consists of:
- Pregnancies: Number of times pregnant.
- Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age (years)
- Outcome: Class variable (0 or 1) 268 of 768 are 1, the others are 0

## table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Machine Learning Prediction](#machine-learning-prediction)
  - [Install dependencies](#install-dependencies)
  - [Data Collection and processing](#data-collection-and-processing)
  - [Data standardization](#data-standardization)

## Project Overview

## Dataset

## Machine Learning Prediction
### Install dependencies
Set up a basic environment for building a machine-learning model, specifically using Support Vector Machines (SVMs).

### Data Collection and processing
- Load and explore a diabetes dataset,
- Separate it into features and labels
- Print out the column names for further use.
It provides a foundation for further data analysis and model development, specifically for a classification task predicting whether a person is diabetic or not based on the provided features.

### Data standardization
- Standardize the feature values in the dataset, meaning you scale them to have a mean of 0 and a standard deviation of 1. This is an important preprocessing step in machine learning, particularly for algorithms like Support Vector Machines (SVMs), which are sensitive to the scale of input data.
- Standardized data is then re-assigned to the feature matrix X, ready for model training.
