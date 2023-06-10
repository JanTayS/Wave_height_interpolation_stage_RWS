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
    df = pd.read_csv('model_dataset.csv')
    print(df.shape)
    target_variable = 'target'
    variables = ['Hm0','WS10','PQFF10']

    loaded_model = keras.models.load_model('model.h5')

    test_model = regression_model(df,target_variable,variables)
    test_model.plot_performance(loaded_model)