# Boston Housing Price Prediction

This project aims to predict the median value of owner-occupied homes in the Boston area based on various features. It explores data cleaning, exploratory data analysis, feature selection, and model training using different regression techniques, including a deep learning model.

## Project Structure

-   **`HousingData.csv`**: The dataset used for training and evaluation.
-   **Jupyter Notebook (`.ipynb` file)**: Contains the Python code for data loading, cleaning, analysis, model training, and evaluation.

## Dataset

The dataset contains information about various aspects of residential areas in the Boston metropolitan area. Key features include:

-   `CRIM`: Per capita crime rate by town
-   `ZN`: Proportion of residential land zoned for lots over 25,000 sq.ft.
-   `INDUS`: Proportion of non-retail business acres per town
-   `CHAS`: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
-   `NOX`: Nitric oxides concentration (parts per 10 million)
-   `RM`: Average number of rooms per dwelling
-   `AGE`: Proportion of owner-occupied units built prior to 1940
-   `DIS`: Weighted distances to five Boston employment centres
-   `RAD`: Index of accessibility to radial highways
-   `TAX`: Full-value property-tax rate per $10,000
-   `PTRATIO`: Pupil-teacher ratio by town
-   `B`: 1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
-   `LSTAT`: % lower status of the population
-   `MEDV`: Median value of owner-occupied homes in $1000s (Target Variable)

## Project Steps

1.  **Data Loading and Initial Exploration**: Load the dataset and perform initial checks on its structure, data types, and summary statistics.
2.  **Data Cleaning**: Handle missing values and outliers in the dataset using appropriate techniques (e.g., mean/median imputation, Isolation Forest).
3.  **Exploratory Data Analysis (EDA)**: Visualize the data distribution, relationships between features, and correlation with the target variable using box plots, scatter plots, and a correlation heatmap.
4.  **Feature Selection**: Analyze feature correlations to identify and potentially remove highly correlated features to avoid multicollinearity.
5.  **Data Splitting**: Split the data into training and testing sets for model development and evaluation.
6.  **Model Training and Evaluation (Traditional Models)**:
    -   Train various regression models (Linear Regression, Decision Tree, Random Forest, etc.).
    -   Evaluate their performance using metrics like Root Mean Squared Error (RMSE) and R-squared.
7.  **Hyperparameter Tuning (Random Forest)**: Use GridSearchCV to find the best hyperparameters for the Random Forest model to optimize its performance.
8.  **Deep Learning Model**:
    -   Build a simple feedforward neural network using TensorFlow/Keras.
    -   Train and evaluate the deep learning model.
    -   Visualize the training and validation loss curves.
9.  **Model Comparison**: Compare the performance of the traditional models and the deep learning model to identify the best performing model.

## Dependencies

-   pandas
-   numpy
-   matplotlib
-   seaborn
-   scikit-learn
-   tensorflow

You can install these dependencies using pip:

## How to Run

1.  Download the `HousingData.csv` file.
2.  Open the Jupyter Notebook (`.ipynb` file) in a Jupyter environment (like Google Colab or Jupyter Notebook).
3.  Run the code cells sequentially to execute the project steps.

## Results

The notebook will display the performance metrics (RMSE and R-squared) for each trained model. The hyperparameter tuning section will show the best parameters found for the Random Forest model and its performance on the test set. The deep learning section will show the training progress and the final evaluation metrics.

## Conclusion

This project provides a comprehensive workflow for predicting housing prices, from data preprocessing to model training and evaluation. The results indicate the effectiveness of different modeling techniques on this dataset.