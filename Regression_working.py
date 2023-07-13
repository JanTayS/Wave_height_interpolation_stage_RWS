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



class regression_model:
    def __init__(self, data_file, target_variable):
        self.datetime_df = pd.read_csv(data_file)
        self.df = self.datetime_df.drop('datetime', axis=1, inplace=False)
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
    
    def train_test_split(self, show_stats=False):
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
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mean_absolute_error',
                        optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def regression(self, show_progress=1, show_plots=True):
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

        self.model_info['model'] = regression_model
        self.model_info['history'] = history
        self.model_info['test_restult'] = regression_model.evaluate(
        self.test_features[self.input_variables],
        self.test_labels, verbose=1)

        if show_plots:
            self.plot_performance(regression_model, predictors)
            self.plot_loss(history)
        
        return regression_model
    
    def hp_tuning(self):
        pass
        # return best_hp

    def plot_performance(self, model, save=False):
        test_predictions = model.predict(self.test_features[self.input_variables]).flatten()

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
    def plot_over_time(self, models, save = False):
        count = 1        
        for model in models:
            predictions = model.predict(self.df[self.input_variables]).flatten()
            plt.plot(self.datetime_df['datetime'][:5000], predictions[:5000], label=f'Predictions_{count}')
            count+=1
        plt.plot(self.datetime_df['datetime'][:5000], self.df['target'][:5000], label='True Values')
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


if __name__ == '__main__':
    # Read in the data
    location = 'wave_height'
    # data_file = f'model_dataset_Hm0_{location}.csv'
    data_file = f'model_datasets/model_dataset2.csv'
    target_variable = 'target'

    DNN_model = regression_model(data_file,target_variable)
    print(DNN_model.df.shape)
    
    print('Multiple_dnn')
    DNN_model.regression()
    class_file = data_file.split('.')[0]
    if os.path.exists(f'{class_file}.pkl'):
            i = 1
            while os.path.exists(f"{class_file}_{i}.pkl"):
                i += 1
            class_file = f"class_folder/{class_file}_{i}.pkl"
    with open(class_file, 'wb') as file:
        pickle.dump(DNN_model, file)

    save_model = True
    if save_model:
        model_name = f'model_{location}'
        if os.path.exists(model_name):
                i = 1
                while os.path.exists(f"{model_name}_{i}"):
                    i += 1
                model_name = f"{model_name}_{i}"

        DNN_model.model_info['model'].save(f'models/{model_name}')

    