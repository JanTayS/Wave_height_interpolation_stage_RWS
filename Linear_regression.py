import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# Read the training and test datasets
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

def get_predictors(df, location='K131', variable='Hm0', cutoff=0.7):
    potential_predictors = []
    for column in df.columns:
        # Find the column that corresponds to the specified location and variable
        if location in column and variable in column:
            location_variable = column
            continue
        # Find potential predictor columns that have the specified variable
        if variable in column:
            potential_predictors.append(column)
    # Calculate the correlations between the potential predictors and the location_variable
    correlations = df[potential_predictors + [location_variable]].corr()[location_variable]
    # Filter the correlations to select only highly correlated values
    high_corr_values = correlations[correlations > cutoff].dropna(how='all')
    # Get the list of predictor columns
    predictors = high_corr_values.index.tolist()
    # Remove the location_variable from the predictors list
    predictors.remove(location_variable)
    return predictors, location_variable

def get_locations(df):
    locations = set()
    for column in df.columns:
        if '_' in column and 'Hm0' in column:
            location = column.split('_')[1]
            locations.add(location)
    return sorted(list(locations))

def find_empty_columns(df):
    empty_columns = []
    for column in df.columns:
        missing_values = df[column].isnull().sum()
        if missing_values == df.shape[0]:
            empty_columns.append(column)
    return empty_columns

def linear_regression(df_train, df_test, location='K131', variable='Hm0'):
    # Get the predictors and location_variable
    predictors, location_variable = get_predictors(df_test, location, variable)
    variables = predictors + [location_variable]
    all_variables = []

    # Calculate and print the number of missing values for each column
    for column in variables:
        missing_values = df_test[column].isnull().sum()
        # print(f"Column '{column}' has {missing_values} missing value(s).")
        # Keep columns with fewer missing values
        if missing_values < df_test.shape[0]:
            all_variables.append(column)
        else:
            predictors.remove(column)

    # Remove rows with missing values in the selected variables
    df_train.dropna(subset=all_variables, inplace=True)
    df_test.dropna(subset=all_variables, inplace=True)

    # Prepare training data
    X = df_train[predictors]
    y = df_train[location_variable]

    # Create and train the linear regression model
    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    # Print the intercept, coefficients, and R-squared score of the model
    print("Intercept:", regr.intercept_)
    print("Coefficients:", regr.coef_)
    print("R-squared:", regr.score(X, y))

    # Prepare test data
    X_test = df_test[predictors]
    y_test = df_test[location_variable]
    x_axis = pd.to_datetime(df_test['datetime'])  # Replace 'datetime_column' with the actual name of the datetime column

    # Predict the output variable for the test set
    y_pred = regr.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    rmse = np.sqrt(mse)
    print("Root Mean Squared Error:", rmse)

    r2 = r2_score(y_test, y_pred)
    print("R-squared:", r2)

    # Calculate the confidence interval
    X_test_with_intercept = sm.add_constant(X_test)  # Add constant term to X_test for the intercept
    model = sm.OLS(y_test, X_test_with_intercept)
    result = model.fit()
    pred = result.get_prediction(X_test_with_intercept)
    conf_interval = pred.conf_int(0.01)  # Calculate the 99% confidence interval

    # Access lower and upper bounds of the confidence interval
    lower_bound = conf_interval[:, 0]
    upper_bound = conf_interval[:, 1]

    # Plot the actual, predicted values, and confidence interval
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, y_test, color='b', label='Actual')  # Use x_axis on the x-axis
    plt.plot(x_axis, y_pred, color='r', label='Predicted')  # Use x_axis on the x-axis
    plt.fill_between(x_axis, lower_bound, upper_bound, color='gray', alpha=0.3, label='Confidence Interval')  # Fill the confidence interval
    plt.xlabel('Datetime')
    plt.ylabel(f'{variable}_{location}')
    plt.title('Actual vs. Predicted Values')
    plt.legend()
    plt.show()


empty_columns = find_empty_columns(df_train) + find_empty_columns(df_test)
df_train.drop(empty_columns, axis=1, inplace=True)
df_test.drop(empty_columns, axis=1, inplace=True)

locations = get_locations(df_test)
for location in locations:
    df_train_copy = df_train.copy()
    df_test_copy = df_test.copy()
    print(location)
    linear_regression(df_train_copy, df_test_copy, location=location)


