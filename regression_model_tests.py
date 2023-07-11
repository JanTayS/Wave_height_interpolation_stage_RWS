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
    location = 'Hm0_AWG1'
    # data_file = 'model_datasets/version_3/model_dataset_test.csv'
    data_file = f'model_datasets/version_3/model_dataset_{location}.csv'
    # data_file = 'model_datasets/version_3/model_dataset_all.csv'
    
    target_variable = 'target'

    # models = ['models\Linear_models\dataset_Hm0_K131_1']
    # model = 'models\Linear_models\dataset_Hm0_K131_1'
    models = ['models/Final_model/model_0']
    model = 'models/Final_model/model_0' 

    
    model_test = RegressionModel(data_file,target_variable)
    # model_test.plot_over_time(models, prediction_count=5000)
    model_test.plot_performance(model)
    model_test.show_metrics(models)

    