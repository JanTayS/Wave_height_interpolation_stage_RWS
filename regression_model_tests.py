import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
from Regression_working import regression_model
import os

if __name__ == '__main__':
    # Read in the data
    location = 'Hm0_A121'
    # data_file = 'model_datasets/version_3/model_dataset_training.csv'
    data_file = f'model_datasets/version_3/model_dataset_{location}.csv'
    
    target_variable = 'target'

    # models = ['models\model_wind_dir','models\model_wind_xy', 'models/model_just_wave_height']
    models = ['models\model_wind_xy', 'models/model_just_wave_height']
    
    model_test = regression_model(data_file,target_variable)
    model_test.plot_over_time(models, prediction_count=10000)
    # model_test.plot_performance(models)
    # model_test.show_metrics(models)

    