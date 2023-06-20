import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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



class regression_model:
    def __init__(self, data_file, target_variable='target'):
        self.datetime_df = pd.read_csv(data_file, engine='pyarrow')
        self.df = self.datetime_df.drop('datetime', axis=1, inplace=False)
        self.target_variable = target_variable
        self.input_variables = self.get_all_inputs()
        self.model_info = {}
        (
            self.train_features,
            self.test_features,
            self.train_labels,
            self.test_labels
        ) = self.train_test_split()

    def get_all_inputs(self):
        all_variables = self.df.columns
        input_variables = [var for var in all_variables if var != self.target_variable]
        return input_variables
    
    def get_model_predictors(self, predictor_variables):
        predictors = []
        for predictor_variable in predictor_variables:
            for column in self.df.columns:
                if predictor_variable in column:
                    predictors.append(column)
        return predictors
    
    def read_model_dir(self, model_dir):
        model_path = os.path.join(model_dir, 'model')
        model = keras.models.load_model(model_path)
        predictors_path = os.path.join(model_dir, 'predictors.json')
        with open(predictors_path, 'r') as json_file:
            predictors = json.load(json_file)
        return model, predictors
    
    def train_test_split(self):
        train_dataset = self.df.sample(frac=0.8, random_state=0)
        test_dataset = self.df.drop(train_dataset.index)

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = train_features.pop(self.target_variable)
        test_labels = test_features.pop(self.target_variable)

        # print(train_features.isna().sum().to_string())

        return train_features, test_features, train_labels, test_labels
    
    def build_and_compile_model(self, normalizer):
        model = keras.Sequential([
            normalizer,
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mean_absolute_error',
                        optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def regression(self, predictor_variables = ['Hm0','hour_avg','wind', 'PQFF10','distance'], show_progress=1, show_plots=True):
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

        save_count = 0
        save_directory = f'models/model_{save_count}'
        while os.path.exists(save_directory):
            save_count += 1
            save_directory = f'models/model_{save_count}'        

        model_file_path = os.path.join(save_directory, 'model')
        regression_model.save(model_file_path)

        # Save history
        history_file_path = os.path.join(save_directory, 'history.pkl')
        with open(history_file_path, 'wb') as file:
            pickle.dump(history, file)

        # Save evaluation results
        evaluation_file_path = os.path.join(save_directory, 'evaluation.pkl')
        evaluation = regression_model.evaluate(self.test_features[predictors], self.test_labels, verbose=1)
        with open(evaluation_file_path, 'wb') as file:
            pickle.dump(evaluation, file)

        predictors_file_path = os.path.join(save_directory, 'predictors.json')
        # Save predictors
        with open(predictors_file_path, 'w') as file:
            json.dump(predictors, file)

        if show_plots:
            self.plot_performance(regression_model, predictors)
            self.plot_loss(history)
        
    
    def hp_tuning(self):
        pass
        # return best_hp

    def plot_performance(self, model, predictors, save=False):
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
            model, predictors = self.read_model_dir(model_path)
            predictions.append(model.predict(self.df[predictors][-prediction_count:]).flatten())
            model_names.append(model_path.split('/')[-1])
            for column in self.df[predictors].columns:
                print(column)
            break
        for prediction, model_name in zip(predictions,model_names):
            plt.plot(self.datetime_df['datetime'][-prediction_count:], prediction, label=model_name)
        plt.plot(self.datetime_df['datetime'][-prediction_count:], self.df['target'][-prediction_count:], label='True Values')
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
            model, predictors = self.read_model_dir(model_path)
            predictions = model.predict(self.df[predictors]).flatten()
            model_name = (model_path.split('/')[-1])
            print(model_name)
            predictions = model.predict(self.df[predictors]).flatten()
            print(self.df['target'])
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


if __name__ == '__main__':
    # Read in the data
    location = 'all'
    data_file = 'model_datasets/version_3/model_dataset_training.csv'
    # data_file = 'model_datasets/version_3/model_dataset_Hm0_K141.csv'
    target_variable = 'target'

    DNN_model = regression_model(data_file,target_variable)
    

    # predictor_variables = ['Hm0','hour_avg', 'PQFF10','distance', 'WS10', 'WR10', 'wind']
    predictor_variables = ['Hm0','distance']

    print('Multiple_dnn')
    DNN_model.regression(predictor_variables)



    