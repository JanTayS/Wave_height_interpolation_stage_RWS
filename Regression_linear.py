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



class RegressionModel:
    def __init__(self, data_file, target_variable='target', save_directory=None):
        self.data_file = data_file
        self.df = pd.read_csv(data_file, engine='pyarrow')
        # self.df = pd.read_hdf(data_file, key='data')
        # self.df = self.df.drop('datetime', axis=1, inplace=False)
        self.target_variable = target_variable
        if save_directory == None:
            self.save_directory = self.save_dir()
        else:
            self.save_directory = save_directory
        (
            self.train_features,
            self.test_features,
            self.train_labels,
            self.test_labels
        ) = self.train_test_split()

    def get_model_predictors(self, predictor_variables, excluded_vars=[]):
        predictors = []
        for predictor_variable in predictor_variables:
            for column in self.df.columns:
                if predictor_variable in column:
                    predictors.append(column)
        for excluded_var in excluded_vars:
            for predictor in predictors:
                if excluded_var in predictor:
                    predictors.remove(predictor)
        return predictors
    
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
    
    def save_model(self, regression_model, history, predictor_variables, predictors):
        new_directory = self.data_file.split('_', 3)[-1].split('.', 1)[0]
        # save_directory = self.save_dir(base_dir=self.save_directory, new_dir='model')
        save_directory = self.save_dir(base_dir=self.save_directory, new_dir=new_directory)

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

    def mlp_regression(self, predictor_variables = ['Hm0', 'hour_avg', 'wind', 'PQFF10'], show_progress=1):
        predictors = self.get_model_predictors(predictor_variables)

        # Convert the selected input variables to a numpy array
        input_data = np.array(self.train_features[predictors])
        # Calculate the axis for normalization
        axis = -1 if len(predictors) > 1 else None

        # Create the input normalizer based on the input data
        input_normalizer = layers.Normalization(input_shape=[len(predictors)], axis=axis)
        input_normalizer.adapt(input_data)

        linear_model = tf.keras.Sequential([
        input_normalizer,
        layers.Dense(units=1)
        ])

        linear_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

        early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor for improvement
        patience=5,          # Number of epochs with no improvement
        verbose=0,            # Prints a message when training stops early
        restore_best_weights=True  # Restores the best weights found during training
        )

        history = linear_model.fit(
        self.train_features[predictors],
        self.train_labels,
        epochs=100,
        # Suppress logging.
        verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split = 0.2,
        callbacks=[early_stopping])

        test_results = linear_model.evaluate(
        self.test_features[predictors], self.test_labels, verbose=0)

        self.save_model(linear_model, history, predictor_variables, predictors)

        return test_results
    
    


if __name__ == '__main__':
    # Read in the data
    location = 'all'
    target_variable = 'target'    
    
    data_dir='model_datasets/version_3'
    for file in os.listdir(data_dir):
        if 'Hm0' in file:
            data_file = os.path.join(data_dir, file)
            mlp_model = RegressionModel(data_file,target_variable,'models/Linear_models')
            print(data_file)    
            
            test_results = mlp_model.mlp_regression()
            print(test_results)



    