import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
from Regression_working import regression_model

if __name__ == '__main__':
    # Read in the data
    location = 'NC1'
    data_file = f'model_dataset_Hm0_{location}.csv'
    target_variable = 'target'
    variables = ['Hm0','WS10','PQFF10']

    loaded_model = keras.models.load_model('model_NC1')

    test_model = regression_model(data_file,target_variable)
    test_model.plot_performance(loaded_model)