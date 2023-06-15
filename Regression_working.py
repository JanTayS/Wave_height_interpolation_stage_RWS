import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
import os
import pickle
from get_variables import get_regression_df
from get_variables import get_regression_df2



class regression_model:
    def __init__(self, data_file, target_variable):
        self.datetime_df = pd.read_csv(data_file)
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
    
    def train_test_split(self, show_stats=False):
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

    def regression(self, show_progress=1, show_plots=True):
        # Convert the selected input variables to a numpy array
        input_data = np.array(self.train_features[self.input_variables])
        # Calculate the axis for normalization
        axis = -1 if len(self.input_variables) > 1 else None

        # Create the input normalizer based on the input data
        input_normalizer = layers.Normalization(input_shape=[len(self.input_variables)], axis=axis)
        input_normalizer.adapt(input_data)
        
        regression_model = self.build_and_compile_model(input_normalizer)
        
        # Define the EarlyStopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor for improvement
        patience=5,          # Number of epochs with no improvement
        verbose=0,            # Prints a message when training stops early
        restore_best_weights=True  # Restores the best weights found during training
        )

        history = regression_model.fit(
        self.train_features[self.input_variables],
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
            self.plot_performance(regression_model)
            self.plot_loss(history)
        
        return regression_model
    
    def hp_tuning(self):
        pass
        # return best_hp

    def plot_performance(self, model, save=False):
        test_predictions = model.predict(self.test_features[self.input_variables]).flatten()

        a = plt.axes(aspect='equal')
        plt.scatter(self.test_labels, test_predictions)
        plt.xlabel(f'True Values [Hm0]')
        plt.ylabel(f'Predictions [Hm0]')
        plt.axis('auto')  # Automatically adjust the axis limits to fit the data

        # Calculate the limits based on the data
        x_limits = plt.xlim()
        y_limits = plt.ylim()
        lims = [min(x_limits[0], y_limits[0]), max(x_limits[1], y_limits[1])]

        # Plot the diagonal line
        _ = plt.plot(lims, lims)
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
        plt.ylabel('Value')
        plt.title('Predictions vs True Values')
        plt.legend()
        plt.show()



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

    