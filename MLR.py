import pandas as pd
import os
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, HalvingGridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFECV, RFE
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVR
from mlxtend.feature_selection import SequentialFeatureSelector as mxsfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import statsmodels.api as sm
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

def select_df_columns(df, variables = ['Hm0','WS10','wind_x','wind_y', 'PQFF10', 'hour_avg']):
    columns = []
    for column in df.columns:
        for variable in variables:
            if variable == 'Hm0' or variable == 'WS10':
                if variable in column and not 'hour_avg' in column:
                    columns.append(column)
            else:
                if variable in column:
                    columns.append(column)
    return columns

def select_features_univariate(X, y, k=5):
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return selected_features

def select_features_lasso(X, y):
    lasso = LassoCV(cv=5, eps=1e-4, max_iter=5000, n_jobs=-1, verbose=1).fit(X, y)
    coef = pd.Series(lasso.coef_, index = X.columns)
    selected_features = coef[coef != 0].index
    return selected_features

def select_features_sfs(X, y):
    sfs = SFS(LinearRegression(), 
           n_features_to_select = 'auto', 
           direction='forward', 
           scoring='neg_mean_absolute_error',
           cv=5,
           n_jobs=-1,
           tol=0.2)
    sfs.fit(X, y)
    selected_features = list(sfs.feature_names_in_)
    return selected_features

def select_features_sfs_mlxtend(X, y):
    sfs = mxsfs(estimator=LinearRegression(),
                                    k_features=10,
                                    forward=True,
                                    scoring='r2',
                                    cv=5,
                                    n_jobs=-1,
                                    verbose=1)
    sfs.fit(X, y)
    fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')

    plt.ylim([0.8, 1])
    plt.title('Sequential Forward Selection (w. StdDev)')
    plt.grid()
    plt.show()
    selected_features = list(sfs.feature_names_in_)
    return selected_features

def select_features_rfecv(X, y):
    selector = RFECV(LinearRegression(), step=1, cv=5, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)
    selector.fit(X, y)
    selected_features = X.columns[selector.support_]
    return selected_features

def choose_n_components(X, plot=False):
    # Fit PCA with a large number of components
    pca = PCA().fit(X)

    # Plot the explained variance ratio
    if plot:
        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.show()

    # Choose the number of components such that 95% of the variance is retained
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.where(cumsum > 0.99)[0][0] + 1
    print(n_components)
    return n_components

def perform_pca(X):
    n_components = choose_n_components(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    return X_pca_df

def create_train_test(file_path, feature_selection_func=None, PCA=False, **kwargs):
    df = pd.read_csv(file_path, engine='pyarrow')

    X = df.drop(['datetime', 'target'], axis=1, inplace=False)
    y = df['target']

    # Normalize the features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)  # Convert to DataFrame
    
    if PCA:
        X_normalized_df = perform_pca(X_normalized_df)

    if feature_selection_func:
        selected_feature_names = feature_selection_func(X_normalized_df, y, **kwargs)  # Use DataFrame
        X_normalized_df = X_normalized_df[selected_feature_names]

    X_normalized_df = sm.add_constant(X_normalized_df)

    X_train, X_test, y_train, y_test = train_test_split(X_normalized_df, y, test_size=0.7, random_state=1)

    return X_train, X_test, y_train, y_test

def test_model(model, X_test, y_test):
    X_test = sm.add_constant(X_test)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print('Mean Absolute Error:', mae)
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', rmse)
    print('R-squared:', r2)
    return [mae, mse, rmse, r2]

def MLR(X_train, y_train):
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train)
    results = model.fit()

    print(results.summary())
    return results

def SVR(X_train, y_train, param_grid=None):
    # Create an instance of the LinearSVR model
    svr = LinearSVR(verbose=1, max_iter=10000)

    # Perform parameter tuning if a parameter grid is provided
    if param_grid:
        grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)
        grid_search.fit(X_train, y_train)

        svr = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print('Best Parameters:', best_params)

    # Fit the model on the training data
    svr.fit(X_train, y_train)
    return svr

def create_model(file_path, feature_selection_func=None, PCA=False, model_type=MLR, show_plots=False):
    X_train, X_test, y_train, y_test = create_train_test(file_path, feature_selection_func, PCA)
    model = model_type(X_train, y_train)
    performance = test_model(model, X_test, y_test)
    if show_plots:
        plot_actual_predicted(model, X_test, y_test)

    feature_count = X_train.shape[1]
    return model, performance, feature_count

def plot_actual_predicted(model, X,y):
    X = sm.add_constant(X)
    y_pred = model.predict(X)

    a = plt.axes(aspect='equal')
    plt.scatter(y, y_pred, s=10, edgecolors='black')
    plt.xlabel('True Values [Hm0]')
    plt.ylabel('Predictions [Hm0]')
    plt.axis('auto')  # Automatically adjust the axis limits to fit the data

    # Calculate the limits based on the data
    x_limits = plt.xlim()
    y_limits = plt.ylim()
    lims = [min(x_limits[0], y_limits[0]), max(x_limits[1], y_limits[1])]

    # Set the limits of the x-axis and y-axis
    plt.xlim(0, lims[1])
    plt.ylim(0, lims[1])

    # Plot the diagonal line
    _ = plt.plot(lims, lims, color='red', linewidth=1)

    # Set the figure size to be a square shape
    fig = plt.gcf()
    fig.set_size_inches(6, 6)  # Set the width and height to the same value

    plt.show()

    def plot_over_time(datetime, y, y_preds, save = False):
        for model in y_preds:
            plt.plot(datetime, y_preds[model], label=model)
        plt.plot(datetime, y, label='True Values')
        plt.xlabel('Time')
        plt.ylabel('Hm0')
        plt.title('Predictions vs True Values')
        plt.legend()
        plt.show()
        if save:
            plt.savefig('plot.png')

def create_all(directory='model_datasets/version_5/', feature_selection_func=select_features_lasso):
    # Initialize an empty DataFrame to hold the results
    results = pd.DataFrame(columns=['Location', 'MAE', 'MSE', 'RMSE', 'R2'])
    models_dir = 'models/Final_MLR'
    # Loop over each file in the directory
    for dataset in os.listdir(directory):
        if 'all' not in dataset and 'train' not in dataset and 'test' not in dataset:
            location = re.search('Hm0_(.*).csv', dataset)
            if location:
                location = location.group(1)
            else:
                print(f"No match found in {dataset}. Skipping...")
                continue
            print(location)
            data_path = os.path.join(directory, dataset)
            model, performance, feature_count = create_model(data_path, feature_selection_func)
            mae, mse, rmse, r2 = performance
            # Append the results for this location to the DataFrame
            results = results.append({'Location': location, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'Features' : int(feature_count)}, ignore_index=True)
    
            model_path = os.path.join(models_dir, location)
            # Save the model
            with open(f'{model_path}.pkl', 'wb') as f:
                pickle.dump(model, f)

    # print LaTeX table
    latex_table(results)
    return results

def latex_table(dataframe):
    latex_tabular = dataframe.to_latex(index=False)

    latex_table = f"""
    \\begin{{table}}[htbp]
    \\centering
    \\caption{{Your Caption}}
    \\label{{tab:your_label}}
    {latex_tabular}
    \\end{{table}}
    """
    print(latex_table)


if __name__ == '__main__':
    dataset = 'model_datasets/version_5/model_dataset_Hm0_L91.csv'
    # model = create_model(dataset, select_features_lasso, show_plots=True)
    results = create_all()

    
    
