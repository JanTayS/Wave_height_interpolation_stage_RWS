import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from get_variables import get_regression_df

# Read in the data
df = pd.read_csv('final_data.csv')

class regression_model:
    def __init__(self, df, target_location, input_variables):
        self.df = df
        self.target = target_location
        self.input_variables = input_variables
        self.regression_dataset = get_regression_df(df, target_location, input_variables)
    
    def regression_setup(self, dataset, target_location, show_stats=False):
        train_dataset = dataset.sample(frac=0.8, random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        if show_stats:
            sns.pairplot(train_dataset, diag_kind='kde')
            plt.show()
            print(train_dataset.describe().transpose())

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = train_features.pop(f'Hm0_{target_location}')
        test_labels = test_features.pop(f'Hm0_{target_location}')

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        return train_features, test_features, train_labels, test_labels, normalizer

location = 'K131' # Location for where to predict wave height
single_input = 'Hm0_J61' # Input variable for single input regression

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

pd.set_option('display.max_rows', None)

dataset = get_regression_df(df,location,variables=['Hm0','WS10'])

def regression_setup(dataset, show_stats=False):
    

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    if show_stats:
        sns.pairplot(train_dataset, diag_kind='kde')
        plt.show()
        print(train_dataset.describe().transpose())

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop(f'Hm0_{location}')
    test_labels = test_features.pop(f'Hm0_{location}')

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    return train_features, test_features, train_labels, test_labels, normalizer

train_features, test_features, train_labels, test_labels, normalizer = regression_setup(df,location)

def plot_loss(history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        # plt.ylim([0, 20])
        plt.xlabel('Epoch')
        plt.ylabel('Error [Hm0]')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_input(x, y):
        plt.scatter(train_features[single_input], train_labels, label='Data')
        plt.plot(x, y, color='k', label='Predictions')
        plt.xlabel(single_input)
        plt.ylabel('Hm0')
        plt.legend()
        plt.show()

def single_linear_regression(train_features, test_features, train_labels, test_labels, show_stats=False):

    single_input_variable = np.array(train_features[single_input])

    input_normalizer = layers.Normalization(input_shape=[1,], axis=None)
    input_normalizer.adapt(single_input_variable)

    single_input_model = tf.keras.Sequential([
        input_normalizer,
        layers.Dense(units=1)
    ])

    single_input_model.summary()
    if show_stats:
        print(single_input_model.predict(single_input_variable[:10]))

    # Define the EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Metric to monitor for improvement
    patience=10,          # Number of epochs with no improvement
    verbose=1,            # Prints a message when training stops early
    restore_best_weights=True  # Restores the best weights found during training
    )

    single_input_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

    history = single_input_model.fit(
    train_features['Hm0_K141'],
    train_labels,
    epochs=100,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stopping]  # Pass the EarlyStopping callback
    )

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    if show_stats:
        print(hist.tail())
        plot_loss(history)

    test_results = {}

    test_results['Wave_height_model'] = single_input_model.evaluate(
        test_features['Hm0_K141'],
        test_labels, verbose=0)

    x = tf.linspace(0.0, 700, 700)
    y = single_input_model.predict(x)

    if show_stats:
        plot_input(x, y)

    return 0

# single_linear_regression(train_features, test_features, train_labels, test_labels, show_stats=True)

def multiple_linear_regression(train_features, test_features, train_labels, test_labels, show_stats=False):

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    multiple_input_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])

    multiple_input_model.predict(train_features[:10])

    multiple_input_model.layers[1].kernel

    # Define the EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Metric to monitor for improvement
    patience=10,          # Number of epochs with no improvement
    verbose=1,            # Prints a message when training stops early
    restore_best_weights=True  # Restores the best weights found during training
    )

    multiple_input_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

    history = multiple_input_model.fit(
        train_features,
        train_labels,
        epochs=100,
        # Suppress logging.
        verbose=1,
        # Calculate validation results on 20% of the training data.
        validation_split = 0.2,
        callbacks=[early_stopping])
    if show_stats:
        x = tf.linspace(0.0, 700, 700)
        y = multiple_input_model.predict(x)
        plot_loss(history)

    # test_results['linear_model'] = linear_model.evaluate(
    #     test_features, test_labels, verbose=0)
    return 0

# multiple_linear_regression(train_features, test_features, train_labels, test_labels, show_stats=True)

def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))
    return model

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

dnn_model = build_and_compile_model(normalizer)

dnn_model.summary()

# Define the EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Metric to monitor for improvement
    patience=10,          # Number of epochs with no improvement
    verbose=1,            # Prints a message when training stops early
    restore_best_weights=True  # Restores the best weights found during training
    )

history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=1, 
    epochs=100,
    callbacks=early_stopping)

plot_loss(history)

x = tf.linspace(0.0, 700, 700)
y = dnn_model.predict(x)

plot_input(x, y)


# plot_loss(history)

# test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

# pd.DataFrame(test_results, index=['Mean absolute error [Hm0_K131]']).T

# test_predictions = dnn_model.predict(test_features).flatten()

# a = plt.axes(aspect='equal')
# plt.scatter(test_labels, test_predictions)
# plt.xlabel('True Values [Hm0_K131]')
# plt.ylabel('Predictions [Hm0_K131]')
# lims = [0, 700]
# plt.xlim(lims)
# plt.ylim(lims)
# _ = plt.plot(lims, lims)

# error = test_predictions - test_labels
# plt.hist(error, bins=25)
# plt.xlabel('Prediction Error [Hm0_K131]')
# _ = plt.ylabel('Count')

# dnn_model.save('dnn_model')

# reloaded = tf.keras.models.load_model('dnn_model')

# test_results['reloaded'] = reloaded.evaluate(
#     test_features, test_labels, verbose=0)

# pd.DataFrame(test_results, index=['Mean absolute error [Hm0_K131]']).T




