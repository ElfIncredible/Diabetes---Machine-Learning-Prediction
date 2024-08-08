# Diabetes - Machine Learning Prediction
Develop and deploy a machine learning model to predict whether an individual is diabetic based on various health metrics. Demonstrate the full lifecycle of a machine learning model, from data preprocessing and model training to deployment and real-time prediction.

## table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Machine Learning Prediction](#machine-learning-prediction)
  - [Install dependencies](#install-dependencies)
  - [Data Collection and processing](#data-collection-and-processing)
  - [Data standardization](#data-standardization)
  - [Training and Testing the data](#training-and-testing-the-data)
  - [Model Training - SupportVectorMachine](#model-training---supportvectormachine)
  - [Model evaluation](#model-evaluation)
  - [Making predictions](#making-predictions)
  - [Saving the trained model](#saving-the-trained-model)

## Project Overview
This project develops a machine learning system to predict diabetes status based on health metrics. The goals and objectives are:

**Goals:**
- **Accurate Prediction:** Create a model that accurately predicts whether an individual is diabetic or non-diabetic based on provided health metrics.
- **Model Deployment:** Develop a system that allows for easy saving, loading, and reuse of the trained model and preprocessing scaler.


## Dataset
This [dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset) is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

Information about dataset attributes:
- Pregnancies: To express the Number of pregnancies
- Glucose: To express the Glucose level in blood
- BloodPressure: To express the Blood pressure measurement
- SkinThickness: To express the thickness of the skin
- Insulin: To express the Insulin level in blood
- BMI: To express the Body mass index
- DiabetesPedigreeFunction: To express the Diabetes percentage
- Age: To express the age
- Outcome: To express the final result 1 is Yes and 0 is No

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

### Training and Testing the data
- Split the standardized feature matrix X and the label vector Y into training and testing subsets. The training set (X_train and y_train) contains 80% of the data, while the testing set (X_test and y_test) contains 20%.
- Stratify the split, ensuring that the class distribution remains consistent across the training and testing sets.
- Print the shapes of the original, training, and testing datasets to verify the split.

### Model Training - SupportVectorMachine
- Create an SVM classifier with a linear kernel and then trains it on the training data. The model learns to classify the data by finding the best linear decision boundary that separates the classes in the training set.
- Once trained, this classifier can be used to predict the class of new, unseen data.

### Model evaluation
- Assess how well the trained SVM classifier performs on both the training and testing datasets by calculating and printing the accuracy scores for each.
- The training accuracy (X_train_accuracy) indicates how well the model has learned the training data, while the testing accuracy (X_test_accuracy) shows how well the model generalizes to unseen data.
- High accuracy on both sets suggests a well-performing model, whereas a significant difference might indicate overfitting or underfitting.

### Making predictions
- The system processes a single instance of health data, standardizes it, and uses a previously trained SVM classifier to predict whether the person is diabetic or not.
- The result is printed as either "The person is diabetic" or "The person is not diabetic" based on the prediction.
This could be part of a diagnostic tool in a healthcare application.

### Saving the trained model
- Save a trained machine learning model and its associated data scaler to files using pickle
- Load them back for future use.
- Standardize new input data using the saved scaler and make predictions using the saved model, providing a reusable approach to deploying machine learning models.
- The final output is a prediction of whether a person with given health metrics is diabetic or not.
