import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor



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
    df_test.dropna(subset=predictors, inplace=True)
    # df_test.dropna(subset=all_variables, inplace=True)

    # Prepare training data
    X = df_train[predictors]
    y = df_train[location_variable]

    # Prepare test data
    X_test = df_test[predictors]
    y_test = df_test[location_variable]
    x_axis = pd.to_datetime(df_test['datetime'])  # Replace 'datetime_column' with the actual name of the datetime column

    # Add constant term to X for the intercept
    X_with_intercept = sm.add_constant(X)

    # Create and train the linear regression model
    regr = sm.OLS(y, X_with_intercept)
    result = regr.fit()

    # Print the intercept, coefficients, and R-squared score of the model
    print("Intercept:", result.params[0])
    print("Coefficients:", result.params[1:])
    print("R-squared:", result.rsquared)

    # Prepare test data
    X_test = df_test[predictors]

    # Add constant term to X_test for the intercept
    X_test_with_intercept = sm.add_constant(X_test)

    # Predict the output variable for the test set
    y_pred = result.predict(X_test_with_intercept)

    # Calculate the confidence interval
    alpha = 0.01
    pred = result.get_prediction(X_test_with_intercept)
    confidence_interval = pred.conf_int(alpha)

    # Plot y_pred against y_test
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Plotting the diagonal line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()
   
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot y_test as a line graph
    ax.plot(x_axis, y_test, label='Actual')

    # Plot y_pred as a line graph
    ax.plot(x_axis, y_pred, label='Predicted')

    # Plot the confidence interval
    ax.fill_between(x_axis, confidence_interval[:, 0], confidence_interval[:, 1], color='gray', alpha=0.3, label=f'Confidence Interval {chr(945)} = {str(alpha)}')

    # Set labels and title
    ax.set_xlabel('datetime')
    ax.set_ylabel(f'{variable}_{location}')
    ax.set_title('Actual vs Predicted with Confidence Interval')

    # Display legend
    ax.legend()

    # Show the plot
    plt.show()

def svr_regression(df_train, df_test, location='K131', variable='Hm0'):
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
    df_test.dropna(subset=predictors, inplace=True)

    # Prepare training data
    X = df_train[predictors]
    y = df_train[location_variable]

    # Prepare test data
    X_test = df_test[predictors]
    y_test = df_test[location_variable]
    x_axis = pd.to_datetime(df_test['datetime'])  # Replace 'datetime_column' with the actual name of the datetime column

    # Create a StandardScaler object
    scaler = StandardScaler()

    # Apply scaling to the training data
    X_scaled = scaler.fit_transform(X)

    # Apply scaling to the test data
    X_test_scaled = scaler.transform(X_test)

    # Create and train the SVR model
    regr = SVR()  # Set verbose and max_iter for the SVR model
    regr.fit(X, y)

    # Predict the output variable for the test set
    y_pred = regr.predict(X_test)

    # Calculate the R-squared score
    # r2 = r2_score(y_test, y_pred)
    # print("R-squared:", r2)

    # Plot y_pred against y_test
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Plotting the diagonal line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot y_test as a line graph
    ax.plot(x_axis, y_test, label='Actual')

    # Plot y_pred as a line graph
    ax.plot(x_axis, y_pred, label='Predicted')

    # Set labels and title
    ax.set_xlabel('datetime')
    ax.set_ylabel(f'{variable}_{location}')
    ax.set_title('Actual vs Predicted')

    # Display legend
    ax.legend()

    # Show the plot
    plt.show()


def mlp_regression(df_train, df_test, location='K131', variable='Hm0'):
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
    df_test.dropna(subset=predictors, inplace=True)

    # Prepare training data
    X_train = df_train[predictors]
    y_train = df_train[location_variable]

    # Prepare test data
    X_test = df_test[predictors]
    y_test = df_test[location_variable]
    x_axis = pd.to_datetime(df_test['datetime'])  # Replace 'datetime_column' with the actual name of the datetime column

    # Create and train the MLP model
    model = MLPRegressor(hidden_layer_sizes=(100, 100),verbose=1,max_iter=2000)  # Adjust the hidden_layer_sizes as needed
    model.fit(X_train, y_train)

    # Predict the output variable for the test set
    y_pred = model.predict(X_test)

    # Find the indices of non-missing values in both y_test and y_pred
    non_missing_indices = np.logical_and(~np.isnan(y_test), ~np.isnan(y_pred))

    # Filter y_pred and y_test based on non-missing indices
    y_pred_non_missing = y_pred[non_missing_indices]
    y_test_non_missing = y_test[non_missing_indices]

    # Calculate evaluation metrics for non-missing values
    mse = mean_squared_error(y_test_non_missing, y_pred_non_missing)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_non_missing, y_pred_non_missing)
    r2 = r2_score(y_test_non_missing, y_pred_non_missing)

    print("Mean Squared Error (non-missing values):", mse)
    print("Root Mean Squared Error (non-missing values):", rmse)
    print("Mean Absolute Error (non-missing values):", mae)
    print("R2 Score (non-missing values):", r2)

    # Plot y_pred against y_test
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Plotting the diagonal line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot y_test as a line graph
    ax.plot(x_axis, y_test, label='Actual')

    # Plot y_pred as a line graph
    ax.plot(x_axis, y_pred, label='Predicted')

    # Set labels and title
    ax.set_xlabel('datetime')
    ax.set_ylabel(f'{variable}_{location}')
    ax.set_title('Actual vs Predicted')

    # Display legend
    ax.legend()

    # Show the plot
    plt.show()

def decision_tree_regression(df_train, df_test, location='K131', variable='Hm0'):
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
    df_test.dropna(subset=predictors, inplace=True)
    # df_test.dropna(subset=all_variables, inplace=True)

    # Prepare training data
    X = df_train[predictors]
    y = df_train[location_variable]

    # Prepare test data
    X_test = df_test[predictors]
    y_test = df_test[location_variable]
    x_axis = pd.to_datetime(df_test['datetime'])  # Replace 'datetime_column' with the actual name of the datetime column

    # Create and train the decision tree regression model
    regr = DecisionTreeRegressor()
    regr.fit(X, y)

    # Predict the output variable for the test set
    y_pred = regr.predict(X_test)

    # Find the indices of non-missing values in both y_test and y_pred
    non_missing_indices = np.logical_and(~np.isnan(y_test), ~np.isnan(y_pred))

    # Filter y_pred and y_test based on non-missing indices
    y_pred_non_missing = y_pred[non_missing_indices]
    y_test_non_missing = y_test[non_missing_indices]

    # Calculate evaluation metrics for non-missing values
    mse = mean_squared_error(y_test_non_missing, y_pred_non_missing)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_non_missing, y_pred_non_missing)
    r2 = r2_score(y_test_non_missing, y_pred_non_missing)

    print("Mean Squared Error (non-missing values):", mse)
    print("Root Mean Squared Error (non-missing values):", rmse)
    print("Mean Absolute Error (non-missing values):", mae)
    print("R2 Score (non-missing values):", r2)

    # Plot y_pred against y_test
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Plotting the diagonal line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.show()

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot y_test as a line graph
    ax.plot(x_axis, y_test, label='Actual')

    # Plot y_pred as a line graph
    ax.plot(x_axis, y_pred, label='Predicted')

    # Set labels and title
    ax.set_xlabel('datetime')
    ax.set_ylabel(f'{variable}_{location}')
    ax.set_title('Actual vs Predicted')

    # Display legend
    ax.legend()

    # Show the plot
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
    
    


