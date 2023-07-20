import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
from MLP import RegressionModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import os
import pickle
from math import sqrt
import re
import json
import statsmodels.api as sm
import glob
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import random
from matplotlib.dates import DateFormatter

if __name__ == '__main__':
    # Read in the data
    location = 'Hm0_L91'
    data_file = 'model_datasets/version_3/model_dataset_test.csv'
    # data_file = f'model_datasets/version_5/model_dataset_all.csv'
    # data_file = 'model_datasets/version_3/model_dataset_all.csv'
    
    target_variable = 'target'

    # models = ['models\Linear_models\dataset_Hm0_K131_1']
    # model = 'models\Linear_models\dataset_Hm0_K131_1'
    # models = ['models/Final_model2/model_0']
    # model = 'models/Final_model2/model_0' 

    
    # model_test = RegressionModel(data_file,target_variable)
    # model_test.plot_over_time(models, prediction_count=5000)
    # model_test.plot_performance(model)
    # model_test.show_metrics(models)


    # with open('models\Final_model2\model_0\history.pkl', 'rb') as file:
    # # Load the contents of the file into a Python object
    #     history = pickle.load(file)
    # model_test.plot_loss(history, save=True)

def get_model_predictors(df, predictor_variables = ['Hm0','WS10','wind_x','wind_y', 'hour_avg']):
        predictors = []
        for predictor_variable in predictor_variables:
            for column in df.columns:
                if predictor_variable == 'Hm0' or predictor_variable == 'WS10':
                    if predictor_variable in column and not 'hour_avg' in column:
                        predictors.append(column)
                else:
                    if predictor_variable in column:
                        predictors.append(column)
        return predictors

def read_model_dir(model_dir = 'models\Final_MLP\model_0'):
        model_path = os.path.join(model_dir, 'model')
        model = keras.models.load_model(model_path)
        predictors_path = os.path.join(model_dir, 'predictor_variables.json')
        with open(predictors_path, 'r') as json_file:
            predictors = json.load(json_file)
        return model, predictors

def metrics(y,y_pred):
    mse = mean_squared_error(y, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("Mean Squared Error: {:.4f}".format(mse))
    print("Root Mean Squared Error: {:.4f}".format(rmse))
    print("Mean Absolute Error: {:.4f}".format(mae))
    print("R^2 Score: {:.4f}".format(r2))

def MLP_predictions(df, model_dir='models\Final_MLP\model_0', show_metrics=False):
    model, predictor_vars = read_model_dir(model_dir)
    predictors = get_model_predictors(df, predictor_vars)
    X = df[predictors]
    y = df['target']
    y_pred = model.predict(X)

    return y_pred
    
def MLR_predictions(df,location='L91', show_metrics=False):
    model_path = f'models\Final_MLR\{location}.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    features = model.model.exog_names
    features.remove('const')
    X = df[features]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = sm.add_constant(X)
    y = df['target']
    y_pred = model.predict(X)
    
    if show_metrics:
        metrics(y,y_pred)

    return y_pred

def plot_over_time(dataset='model_datasets/version_5/model_dataset_Hm0_L91.csv', save=True):
    df = pd.read_csv(dataset, engine='pyarrow')
    y = df['target']
    y_MLP = MLP_predictions(df)
    y_MLR = MLR_predictions(df)
    datetime = df['datetime']
    time_points = 144
    
    plt.plot(datetime[:time_points], y_MLP[:time_points], label='MLP')
    plt.plot(datetime[:time_points], y_MLR[:time_points], label='MLR')
    plt.plot(datetime[:time_points], y[:time_points], label='True Values')
    plt.xlabel('Time')
    plt.ylabel('Hm0')
    plt.title('Predictions vs True Values')
    plt.legend()

    if save:
        plt.savefig('Plots/plot_over_time.png', bbox_inches='tight')
    plt.show()



def plot_over_time(dataset='model_datasets/version_5/model_dataset_Hm0_L91.csv', save=True):
    df = pd.read_csv(dataset, engine='pyarrow')

    # Ensure datetime is in the correct format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    y_MLP = MLP_predictions(df)
    y_MLR = MLR_predictions(df)
    
    # Add predictions to the dataframe
    df['y_MLP'] = y_MLP
    df['y_MLR'] = y_MLR

    # Select a random day
    unique_days = df['datetime'].dt.date.unique()
    random_day = random.choice(unique_days)

    # Filter the DataFrame for the chosen day
    df_day = df[df['datetime'].dt.date == random_day]

    y = df_day['target']
    y_MLP_day = df_day['y_MLP']
    y_MLR_day = df_day['y_MLR']
    datetime = df_day['datetime']

    fig, ax = plt.subplots()
    ax.plot(datetime, y_MLP_day, label='MLP')
    ax.plot(datetime, y_MLR_day, label='MLR')
    ax.plot(datetime, y, label='True Values')

    # Setting the x-axis as time and formatting it
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()  # autoformat the x-axis date label for better fit

    ax.set_xlabel('Time')
    ax.set_ylabel('Hm0')
    ax.set_title('Predictions vs True Values for ' + str(random_day))
    ax.legend()

    # Limit the number of ticks on the x-axis
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))  # Adjust the number of ticks as needed

    if save:
        plt.savefig('Plots/plot_over_time.png', bbox_inches='tight')
    plt.show()

plot_over_time()


def plot_actual_predicted(dataset='model_datasets/version_5/model_dataset_Hm0_L91.csv', model='MLP', save=True):
    if model == 'MLP':
        y_pred = MLP_predictions(dataset)
    elif model == 'MLR':
        y_pred = MLR_predictions(dataset)
    else:
        return print('No model selected')
    df = pd.read_csv(dataset, engine='pyarrow')
    y = df['target']

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
    if save:
        plt.savefig(f'Plots/actual_predicted_{model}.png')
    plt.show()



def plot_actual_predicted_all(directory='model_datasets/version_5/', model='MLP', save=True):
    predictions = []
    true_values = []

    # iterate over all .csv files in the directory
    for file_path in glob.glob(os.path.join(directory, '*.csv')):
        if 'all' in file_path or 'test' in file_path or 'train' in file_path:
            continue
        print(file_path)

        df = pd.read_csv(file_path, engine='pyarrow')
        df = df.sample(frac=0.2, random_state=42)
        y = df['target']
        
        if model == 'MLP':
            y_pred = MLP_predictions(df)
        elif model == 'MLR':
            location = file_path.split('_')[-1].split('.')[0]
            y_pred = MLR_predictions(df, location)
        else:
            return print('No model selected')

        predictions.extend(y_pred)
        true_values.extend(y)

    a = plt.axes(aspect='equal')
    plt.scatter(true_values, predictions, s=10, edgecolors='black')
    plt.xlabel('True Values [Hm0]')
    plt.ylabel('Predictions [Hm0]')
    plt.axis('auto')

    x_limits = plt.xlim()
    y_limits = plt.ylim()
    lims = [min(x_limits[0], y_limits[0]), max(x_limits[1], y_limits[1])]

    plt.xlim(0, lims[1])
    plt.ylim(0, lims[1])

    _ = plt.plot(lims, lims, color='red', linewidth=1)

    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    if save:
        plt.savefig(f'Plots/actual_predicted_{model}_all.png')
    plt.show()

def plot_actual_predicted_all2(directory='model_datasets/version_5/', save=True):
    predictions_MLP = []
    predictions_MLR = []
    true_values_MLP = []
    true_values_MLR = []

    # iterate over all .csv files in the directory
    for file_path in glob.glob(os.path.join(directory, '*.csv')):
        if 'all' in file_path or 'test' in file_path or 'train' in file_path:
            continue
        print(file_path)

        df = pd.read_csv(file_path, engine='pyarrow')
        df = df.sample(frac=0.2, random_state=42)
        y = df['target']
        
        y_pred_MLP = MLP_predictions(df)
        y_pred_MLR = MLR_predictions(df, file_path.split('_')[-1].split('.')[0])

        predictions_MLP.extend(y_pred_MLP)
        true_values_MLP.extend(y)

        predictions_MLR.extend(y_pred_MLR)
        true_values_MLR.extend(y)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Change here

    for ax, preds, trues, title in zip(axs, [predictions_MLP, predictions_MLR], 
                                       [true_values_MLP, true_values_MLR], 
                                       ['MLP', 'MLR']):
        ax.scatter(trues, preds, s=10, edgecolors='black')
        lims = [0,  # min of both axes set to 0
                np.max([ax.get_xlim(), ax.get_ylim()])]  # max of both axes
        ax.plot(lims, lims, 'r')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')  # Make the plot square
        ax.set_xlabel('True Values [Hm0]')
        ax.set_ylabel('Predictions [Hm0]')
        ax.set_title(title)

    fig.tight_layout()
    if save:
        plt.savefig(f'Plots/actual_predicted_all.png')
    plt.show()


# plot_actual_predicted_all2()
# plot_actual_predicted_all(model='MLR')


def performance(model_path, directory='model_datasets/version_5/'):
    model = load_model(model_path)
    results = pd.DataFrame(columns=['Location', 'MAE', 'MSE', 'RMSE', 'R2'])
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
            data = pd.read_csv(data_path, engine='pyarrow')
            with open('models\Final_model2\model_0\predictor_variables.json', 'r') as json_file:
                predictor_vars = json.load(json_file)
            predictors = get_model_predictors(data, predictor_vars)
            X = data[predictors]
            y = data['target']

            y_pred = model.predict(X)
            # Calculate mean squared error
            mse = mean_squared_error(y, y_pred)

            rmse = sqrt(mse)

            # Calculate mean absolute error
            mae = mean_absolute_error(y, y_pred)

            # Calculate R^2 score
            r2 = r2_score(y, y_pred)

            results = pd.concat([results, pd.DataFrame({'Location': [location], 'MAE': [mae], 'MSE': [mse], 'RMSE': [rmse], 'R2': [r2]})], ignore_index=True)
            print(results)
        
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

# model = 'models\Final_model2\model_0\model'
# performance(model)