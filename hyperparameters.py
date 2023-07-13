import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import os
import pickle
import json
from keras_tuner.tuners import RandomSearch, Hyperband
from keras_tuner.engine.hyperparameters import HyperParameters



class regression_model:
    def __init__(self, data_file, target_variable='target'):
        # self.datetime_df = pd.read_csv(data_file, engine='pyarrow')
        self.datetime_df = pd.read_hdf(data_file, key='data')
        self.df = self.datetime_df.drop('datetime', axis=1, inplace=False)
        self.target_variable = target_variable
        self.model_info = {}
        (
            self.train_features,
            self.test_features,
            self.train_labels,
            self.test_labels
        ) = self.train_test_split()

    def get_model_predictors(self, predictor_variables):
        predictors = []
        for predictor_variable in predictor_variables:
            for column in self.df.columns:
                if predictor_variable in column:
                    predictors.append(column)
        return predictors
    
    def train_test_split(self):
        train_dataset = self.df.sample(frac=0.8, random_state=0)
        test_dataset = self.df.drop(train_dataset.index)

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = train_features.pop(self.target_variable)
        test_labels = test_features.pop(self.target_variable)

        # print(train_features.isna().sum().to_string())

        return train_features, test_features, train_labels, test_labels
    
    def build_and_compile_model_hp(self, hp):
        model = keras.Sequential()

        # Add the normalization layer
        model.add(layers.Normalization(input_shape=[len(self.input_variables)]))

        # Number of layers
        num_layers = hp.Int('num_layers', 1, 4)
        activation = hp.Choice('activation', values=['relu','leaky_relu'])
        for _ in range(num_layers):
            # Units (neurons) in each layer
            units = hp.Int('units', min_value=32, max_value=512, step=32)
            model.add(layers.Dense(units=units, activation=activation))

        model.add(layers.Dense(1))

        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        optimizer = keras.optimizers.Adam(learning_rate)

        model.compile(
            loss='mean_absolute_error',
            optimizer=optimizer
        )

        return model

    def hyperparameter_tuning(self, predictor_variables=["Hm0", "distance", "WS10", "WR10", "PQFF10", "hour_avg"]):
        predictors = self.get_model_predictors(predictor_variables)
        self.input_variables = predictors        
        input_data = np.array(self.train_features[predictors])

        # Calculate the axis for normalization
        axis = -1 if len(predictors) > 1 else None

        # Create the input normalizer based on the input data
        input_normalizer = layers.Normalization(input_shape=[len(predictors)], axis=axis)
        input_normalizer.adapt(input_data)

        tuner = Hyperband(
            hypermodel=self.build_and_compile_model_hp,
            objective='val_loss',
            max_epochs=20,
            directory='HP_tuning',
            project_name='regression_tuning'
        )

        tuner.search(
            self.train_features[predictors],
            self.train_labels,
            epochs=100,
            validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1,
                restore_best_weights=True
            )]
        )

        # Get the best model and hyperparameters
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        # Save the best hyperparameters
        with open('best_hyperparameters.json', 'w') as f:
            json.dump(best_hyperparameters.get_config(), f)

        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = tuner.hypermodel.build(best_hyperparameters)
        history = model.fit(self.train_features[predictors],
                            self.train_labels, 
                            epochs=100, 
                            validation_split=0.2)

        val_acc_per_epoch = history.history['val_loss']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        hypermodel = tuner.hypermodel.build(best_hyperparameters)

        # Retrain the model
        hypermodel.fit(self.train_features[predictors], self.train_labels, epochs=best_epoch, validation_split=0.2)

        eval_result = hypermodel.evaluate(self.test_features[predictors], self.test_labels)
        print("[test loss, test accuracy]:", eval_result)

        # Save the best model
        hypermodel.save('best_model')


if __name__ == '__main__':
    # Read in the data
    location = 'all'
    data_file = 'model_datasets/version_5/model_dataset_all.csv'
    # data_file = 'model_datasets/version_3/model_dataset_Hm0_K141.csv'
    target_variable = 'target'

    DNN_model = regression_model(data_file,target_variable)

    predictor_variables = ["Hm0", "distance", "WS10", "WR10", "PQFF10", "hour_avg"]

    print('Multiple_dnn')
    # DNN_model.regression(predictor_variables)
    DNN_model.hyperparameter_tuning(predictor_variables)