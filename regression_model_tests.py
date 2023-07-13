import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
from Regression_working import RegressionModel
import os

if __name__ == '__main__':
    # Read in the data
    location = 'D151'
    data_file = f'model_datasets/model_dataset_Hm0_{location}.csv'
    target_variable = 'target'

    models = []
    models_path = 'models'
    for model in os.listdir(models_path):
        model_path = os.path.join(models_path, model)
        models.append(keras.models.load_model(model_path))

    test_model = regression_model(data_file,target_variable)
    test_model.plot_over_time(models,save=True)