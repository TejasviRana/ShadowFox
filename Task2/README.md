# Car Price Prediction Project

This project focuses on predicting the selling price of cars based on various features such as year, kilometers driven, fuel type, transmission, and more. The analysis involves data cleaning, exploratory data analysis (EDA), feature engineering, and training several regression models to find the best predictor.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Steps](#analysis-steps)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to build a model that can accurately predict the selling price of a used car. This involves:
1.  Loading and understanding the dataset.
2.  Handling missing values and duplicates.
3.  Performing exploratory data analysis to understand the data distribution and relationships between features and the target variable.
4.  Engineering new features to potentially improve model performance.
5.  Preprocessing data (e.g., scaling, encoding categorical variables).
6.  Training and evaluating various regression models.
7.  Identifying the best performing model.

## Dataset

The dataset used in this project is `car.csv`. It contains information about various cars, including:
-   `Car_Name`: Name of the car
-   `Year`: Manufacturing year
-   `Selling_Price`: Price at which the car was sold (Target Variable)
-   `Present_Price`: Current ex-showroom price
-   `Kms_Driven`: Total kilometers driven
-   `Fuel_Type`: Type of fuel used (Petrol, Diesel, CNG)
-   `Seller_Type`: Seller type (Dealer, Individual)
-   `Transmission`: Transmission type (Manual, Automatic)
-   `Owner`: Number of previous owners

## Prerequisites

To run this notebook, you will need:
-   Python 3.6 or higher
-   Jupyter Notebook or JupyterLab
-   The following Python libraries:
    -   pandas
    -   numpy
    -   matplotlib
    -   seaborn
    -   scikit-learn
    -   scipy

## Installation

1.  Clone this repository (if applicable):
(If not in a repo, ensure you have the notebook file and `car.csv` in the same directory)

2.  Install the required libraries using pip.

## Usage

1.  Open the Jupyter Notebook/Lab environment.
2.  Navigate to the directory where you saved the notebook file (`.ipynb`).
3.  Open the notebook and run the cells sequentially.

## Analysis Steps

The notebook covers the following key steps:

1.  **Data Loading and Initial Inspection**: Load the `car.csv` file into a pandas DataFrame and perform initial checks (`head()`, `info()`, `describe()`).
2.  **Data Cleaning**: Check for and remove duplicate rows.
3.  **Exploratory Data Analysis (EDA)**: Visualize the distribution of features, relationship between features and `Selling_Price` using plots like histograms, box plots, and pair plots. A correlation heatmap is generated to show relationships between numerical features.
4.  **Feature Engineering**:
    -   Create `Car_Age` from `Year`.
    -   Create interaction terms like `Age_Kms_Interaction`, `Age_Present_Price`, and `Present_Price_Kms_Driven`.
5.  **Data Preprocessing**:
    -   Apply Min-Max and Standard Scaling to numerical features (`Kms_Driven`, `Present_Price`).
    -   Log transform `Kms_Driven` to handle skewness.
    -   Apply Label Encoding to categorical features (`Fuel_Type`, `Seller_Type`, `Transmission`).
6.  **Feature Selection**: Select a subset of features based on correlation with the target variable and domain knowledge.
7.  **Model Training and Evaluation**:
    -   Split the data into training and testing sets (80/20 split).
    -   Train several regression models including Linear Regression, Decision Tree, Random Forest, Poisson Regressor, Lasso, Ridge, and ElasticNet.
    -   Evaluate models using RMSE and R2 score on the test set.
8.  **Model Selection**: Based on the evaluation metrics, the Linear Regression model is considered the best performing among the tested models in this particular run.
9.  **Detailed Evaluation of Best Model**: Evaluate the best model (Linear Regression) on both training and test sets to check for potential overfitting or underfitting.

## Models Evaluated

The following regression models were trained and evaluated:

-   Linear Regression
-   Decision Tree Regressor
-   Random Forest Regressor
-   Poisson Regressor
-   Lasso
-   Ridge
-   ElasticNet

## Results

The performance of the models is summarized in a DataFrame showing RMSE and R2 scores. Linear Regression showed promising results on the initial run. The final evaluation of the chosen Linear Regression model on training and testing sets provides insights into its generalization ability.

## Contributing

Contributions are welcome! If you find issues or have suggestions for improvements, please open an issue or create a pull request.

## License

[Specify the license for your project, e.g., MIT License, Apache 2.0, etc.]