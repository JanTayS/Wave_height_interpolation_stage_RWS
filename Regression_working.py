import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tables
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import os
import pickle
import json
from get_variables import get_regression_df
from get_variables import get_regression_df2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dask import dataframe as dd



class RegressionModel:
    def __init__(self, data_file, target_variable='target'):
        self.df = pd.read_csv(data_file, engine='pyarrow')
        # self.df = pd.read_hdf(data_file, key='data')
        # self.df = self.df.drop('datetime', axis=1, inplace=False)
        self.target_variable = target_variable
        self.save_directory = self.save_dir()
        (
            self.train_features,
            self.test_features,
            self.train_labels,
            self.test_labels
        ) = self.train_test_split()

    def get_model_predictors(self, predictor_variables = ['Hm0','WS10','wind_x','wind_y', 'hour_avg']):
        predictors = []
        for predictor_variable in predictor_variables:
            for column in self.df.columns:
                if predictor_variable == 'Hm0' or predictor_variable == 'WS10':
                    if predictor_variable in column and not 'hour_avg' in column:
                        predictors.append(column)
                else:
                    if predictor_variable in column:
                        predictors.append(column)
        return predictors
    
    def select_df_columns(df, variables ):
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
    
    def read_model_dir(self, model_dir):
        model_path = os.path.join(model_dir, 'model')
        model = keras.models.load_model(model_path)
        predictors_path = os.path.join(model_dir, 'predictor_variables.json')
        with open(predictors_path, 'r') as json_file:
            predictors = json.load(json_file)
        return model, predictors
    
    def save_dir(self, base_dir='models', new_dir='run'):
        save_count = 0
        save_directory = f'{base_dir}/{new_dir}_{save_count}'
        while os.path.exists(save_directory):
            save_count += 1
            save_directory = f'{base_dir}/{new_dir}_{save_count}'
        os.makedirs(save_directory, exist_ok=True)
        return save_directory
    
    def train_test_split(self):
        train_dataset = self.df.sample(frac=0.8, random_state=0)
        test_dataset = self.df.drop(train_dataset.index)

        train_path = os.path.join(self.save_directory, 'train.h5')
        test_path = os.path.join(self.save_directory, 'test.h5')
        train_dataset.to_hdf(train_path, key='data')
        test_dataset.to_hdf(test_path, key='data')

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = train_features.pop(self.target_variable)
        test_labels = test_features.pop(self.target_variable)
        return train_features, test_features, train_labels, test_labels
    
    def build_and_compile_model(self, normalizer):
        model = keras.Sequential([
            normalizer,
            layers.Dense(352, activation='relu'),
            layers.Dense(352, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mean_absolute_error',
                        optimizer=tf.keras.optimizers.Adam(0.0001))
        return model

    def regression(self, predictor_variables = ['Hm0','hour_avg','wind', 'PQFF10','distance'], show_progress=1, show_plots=False):
        predictors = self.get_model_predictors(predictor_variables)

        # Convert the selected input variables to a numpy array
        input_data = np.array(self.train_features[predictors])
        # Calculate the axis for normalization
        axis = -1 if len(predictors) > 1 else None

        # Create the input normalizer based on the input data
        input_normalizer = layers.Normalization(input_shape=[len(predictors)], axis=axis)
        input_normalizer.adapt(input_data)
        
        regression_model = self.build_and_compile_model(input_normalizer)
        
        # Define the EarlyStopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor for improvement
        patience=10,          # Number of epochs with no improvement
        verbose=0,            # Prints a message when training stops early
        restore_best_weights=True  # Restores the best weights found during training
        )

        history = regression_model.fit(
        self.train_features[predictors],
        self.train_labels,
        epochs=100,
        verbose=show_progress,
        validation_split=0.2,
        callbacks=[early_stopping]  # Pass the EarlyStopping callback
        )

        save_directory = self.save_dir(base_dir=self.save_directory, new_dir='model') 

        model_file_path = os.path.join(save_directory, 'model')
        regression_model.save(model_file_path)

        # Save history
        history_file_path = os.path.join(save_directory, 'history.pkl')
        with open(history_file_path, 'wb') as file:
            pickle.dump(history, file)

        # Save evaluation results
        evaluation_file_path = os.path.join(save_directory, 'val_loss.txt')
        evaluation = regression_model.evaluate(self.test_features[predictors], self.test_labels, verbose=1)
        with open(evaluation_file_path, 'w') as file:
            file.write(str(evaluation))

        predictors_file_path = os.path.join(save_directory, 'predictor_variables.json')
        # Save predictors
        with open(predictors_file_path, 'w') as file:
            json.dump(predictor_variables, file)

        if show_plots:
            self.plot_performance(regression_model, predictors)
            self.plot_loss(history)
        return evaluation

    def plot_performance(self, model, save=False):
        model, predictor_variables = self.read_model_dir(model)
        predictors = self.get_model_predictors(predictor_variables)

        test_predictions = model.predict(self.test_features[predictors]).flatten()
        # test_predictions = model.predict(self.df[predictors]).flatten()

        a = plt.axes(aspect='equal')
        plt.scatter(self.test_labels, test_predictions, s=10, edgecolors='black')
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

        if save:
            plt.savefig('plot.png')

    def plot_over_time(self, models, prediction_count=10000, save = False):
        
        model_names = []
        predictions = []
        for model_path in models:
            model, predictor_variables = self.read_model_dir(model_path)
            predictors = self.get_model_predictors(predictor_variables)
            predictions.append(model.predict(self.df[predictors][-prediction_count:]).flatten())
            model_names.append(model_path.split('/')[-1])
            break
        for prediction, model_name in zip(predictions,model_names):
            plt.plot(self.df['datetime'][-prediction_count:], prediction, label=model_name)
        plt.plot(self.df['datetime'][-prediction_count:], self.df['target'][-prediction_count:], label='True Values')
        plt.xlabel('Time')
        plt.ylabel('Hm0')
        plt.title('Predictions vs True Values')
        plt.legend()
        plt.show()
        if save:
            plt.savefig('plot.png')

    def plot_loss(self, history, save=False):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel(f'Error [Hm0_{self.target_variable}]')
        plt.legend()
        plt.grid(True)
        plt.show()
        if save:
            plt.savefig('plot.png')

    def show_metrics(self, models):
        model_names = []
        predictions = []
        for model_path in models:
            model, predictor_variables = self.read_model_dir(model_path)
            predictors = self.get_model_predictors(predictor_variables)
            predictions = model.predict(self.df[predictors]).flatten()
            model_name = (model_path.split('/')[-1])
            print(model_name)
            predictions = model.predict(self.df[predictors]).flatten()
            print(self.df[predictors])
            # Calculate mean squared error
            mse = mean_squared_error(self.df['target'], predictions)

            # Calculate mean absolute error
            mae = mean_absolute_error(self.df['target'], predictions)

            # Calculate R^2 score
            r2 = r2_score(self.df['target'], predictions)

            # Print the performance metrics
            print("Mean Squared Error: {:.4f}".format(mse))
            print("Mean Absolute Error: {:.4f}".format(mae))
            print("R^2 Score: {:.4f}".format(r2))

    def forward_features(self, base_predictor_variables = ['Hm0','distance'], potential_predictor_variables = [['hour_avg'], ['PQFF10'], ['WS10', 'WR10'], ['wind']]):
        base_val_loss = self.regression(base_predictor_variables,show_plots=False)
        improved_performance = True
        while improved_performance:
            new_val_loss = 0
            performance = []
            for chosen_predictors in potential_predictor_variables:
                predictor_variables = base_predictor_variables + chosen_predictors
                print(predictor_variables)
                val_loss = self.regression(predictor_variables,show_plots=False)
                performance.append([chosen_predictors, val_loss])

            best_new_predictor = []
            new_val_loss = 1000
            for feature in performance:
                if feature[1] < new_val_loss:
                    new_val_loss = feature[1]
                    best_new_predictor = feature[0]
            
            if new_val_loss < base_val_loss:
                base_predictor_variables += best_new_predictor
                potential_predictor_variables.remove(best_new_predictor)
                base_val_loss = new_val_loss
                improved_performance = True
            else: 
                improved_performance = False
                final_features = base_predictor_variables
                final_performace = base_val_loss
        print(final_performace)
        forward_features_path = os.path.join(self.save_directory, 'step_forward_features.json')
        with open(forward_features_path, 'w') as file:
            json.dump(final_features, file)

        return final_features

        

if __name__ == '__main__':
    # Read in the data
    location = 'all'
    data_file = 'model_datasets/version_5/model_dataset_all.csv'
    # data_file = 'model_datasets/version_4/model_dataset_all.h5'
    target_variable = 'target'
    DNN_model = RegressionModel(data_file,target_variable)
    print('Data_loaded')
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    DNN_model.regression(predictor_variables=['Hm0','distance', 'WS10', 'PQFF10', 'WR10', 'hour_avg'])


    