import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from get_variables import get_regression_df
from get_variables import get_regression_df2



class regression_model:
    def __init__(self, df, target_location, variables):
        self.df = df
        self.target_location = target_location
        self.variables = variables # Hm0 (wave height), WS10 (wind speed), WR10 (wind director) 
        self.regression_dataset = get_regression_df2(self.df, self.target_location, self.variables)
        self.all_input_variables = self.get_all_inputs()
        self.test_results = {}
        self.models = {}
        (
            self.train_features,
            self.test_features,
            self.train_labels,
            self.test_labels,
            self.normalizer
        ) = self.regression_setup()

    def get_all_inputs(self):
        all_variables = self.regression_dataset.columns
        all_input_variables = [var for var in all_variables if var != f'Hm0_{self.target_location}']
        return all_input_variables
    
    def regression_setup(self, show_stats=False):
        train_dataset = self.regression_dataset.sample(frac=0.8, random_state=0)
        test_dataset = self.regression_dataset.drop(train_dataset.index)

        if show_stats:
            sns.pairplot(train_dataset, diag_kind='kde')
            plt.show()
            print(train_dataset.describe().transpose())

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        test_features = test_features.dropna(subset=[f'Hm0_{target_location}'])
        train_features = train_features.dropna(subset=[f'Hm0_{target_location}'])
        test_features[self.all_input_variables] = test_features[self.all_input_variables].fillna(test_features[self.all_input_variables].median())
        train_features[self.all_input_variables] = train_features[self.all_input_variables].fillna(train_features[self.all_input_variables].median())

        train_labels = train_features.pop(f'Hm0_{self.target_location}')
        test_labels = test_features.pop(f'Hm0_{self.target_location}')

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        return train_features, test_features, train_labels, test_labels, normalizer
    
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
    
    def plot_loss(self, history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        # plt.ylim([0, 10])
        plt.xlabel('Epoch')
        plt.ylabel(f'Error [Hm0_{target_location}]')
        plt.legend()
        plt.grid(True)
        plt.show()

    def regression(self, input_variables, type='dnn', show_progress=0, show_history=False):

        # Convert the selected input variables to a numpy array
        input_data = np.array(self.train_features[input_variables])

        # Calculate the axis for normalization
        axis = -1 if len(input_variables) > 1 else None

        # Create the input normalizer based on the input data
        input_normalizer = layers.Normalization(input_shape=[len(input_variables)], axis=axis)
        input_normalizer.adapt(input_data)

        if type == 'linear':
            regression_model = tf.keras.Sequential([
                input_normalizer,
                layers.Dense(units=1)
            ])

            regression_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            loss='mean_absolute_error')

        elif type == 'dnn':
            regression_model = self.build_and_compile_model(input_normalizer)

        # Define the EarlyStopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor for improvement
        patience=5,          # Number of epochs with no improvement
        verbose=0,            # Prints a message when training stops early
        restore_best_weights=True  # Restores the best weights found during training
        )

        history = regression_model.fit(
        self.train_features[input_variables],
        self.train_labels,
        epochs=100,
        verbose=show_progress,
        validation_split=0.2,
        callbacks=[early_stopping]  # Pass the EarlyStopping callback
        )

        if show_history:
            self.plot_loss(history)

        test_predictions = regression_model.predict(self.test_features[input_variables]).flatten()

        a = plt.axes(aspect='equal')
        plt.scatter(self.test_labels, test_predictions)
        plt.xlabel(f'True Values [Hm0_{self.target_location}]')
        plt.ylabel(f'Predictions [Hm0_{self.target_location}]')
        plt.plot([0, 700], [0, 700])  # Plotting the diagonal line
        plt.xlim(0, 700)
        plt.ylim(0, 700)
        plt.show()

        self.test_results[f'{input_variables}_{type}_model'] = regression_model.evaluate(
        self.test_features[input_variables],
        self.test_labels, verbose=0)
        self.models[f'{input_variables}_{type}_model'] = regression_model

    def model_performance(self):
        print(pd.DataFrame(self.test_results, index=[f'Mean absolute error [Hm0_{target_location}]']).T)

if __name__ == '__main__':
    # Read in the data
    df = pd.read_csv('final_data.csv')
    target_location = 'NC1'
    variables = ['Hm0','WS10']

    test_model = regression_model(df,target_location,variables)

    # print('Single_linear')
    # test_model.regression(['Hm0_K141'],'linear',1)
    # print('Multiple_linear')
    # test_model.regression(test_model.all_input_variables,'linear',1)
    # print('Single_dnn')
    # test_model.regression(['Hm0_K141'],'dnn',1)
    print('Multiple_dnn')
    test_model.regression(test_model.all_input_variables,'dnn',1)
    test_model.model_performance()









