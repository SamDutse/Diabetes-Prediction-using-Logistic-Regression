**README**

# Diabetes Prediction using Logistic Regression

This project aims to predict whether a patient is diabetic or not based on various health-related features using Logistic Regression. The dataset consists of 768 rows and 9 columns, including features like Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, DiabetesPedigreeFunction, and Age. The "Outcome" column serves as the dependent variable, with values 1 and 0 indicating whether a patient is diabetic or not, respectively.

## Project Overview

The project can be summarized into the following steps:

1. **Data Import and Exploration**: The dataset is loaded using Python's Pandas library, and basic information about the data is checked, including data types and the presence of any missing values.

2. **Data Visualization**: Data visualization is performed using Seaborn and Matplotlib libraries to gain insights into the distribution and relationships between various features.

3. **Data Preprocessing**: The data is split into independent variables (X) and the target variable (y). Then, the dataset is split into training and testing sets using the `train_test_split` function from Scikit-learn. Additionally, feature scaling is applied to standardize the data for better model performance.

4. **Model Training**: Logistic Regression is chosen as the predictive model for this binary classification task. The model is trained on the training data using the `LogisticRegression` class from Scikit-learn.

5. **Model Evaluation**: The trained model is used to predict outcomes on the testing set. The accuracy of the model is calculated using the confusion matrix and accuracy score metrics from Scikit-learn.

## Results

After training and evaluating the Logistic Regression model, it achieved an accuracy of approximately 77.92% on the test set. This means that the model can correctly predict the diabetes status of around 77.92% of patients in the unseen data.

## Project Files

1. `diabetes.csv`: The dataset file containing the necessary information for training and testing the model.

2. `diabetes_prediction.ipynb`: A Jupyter Notebook containing the Python code used for data exploration, preprocessing, model training, and evaluation.

## Libraries Used

1. Pandas: For data manipulation and analysis.
2. Matplotlib and Seaborn: For data visualization.
3. Scikit-learn: For machine learning modeling and evaluation.

## How to Use the Project

1. Install the required libraries mentioned in the Jupyter Notebook.
2. Download the "diabetes.csv" dataset and place it in the appropriate directory.
3. Run the Jupyter Notebook cells in sequential order to perform data exploration, model training, and evaluation.

Please feel free to reach out if you have any questions or feedback!
