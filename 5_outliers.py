import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import os

def remove_outliers_sliding_window(df, window_size=20, sigma=2):
    """
    Removes outliers from a dataframe column using a sliding window approach.

    Args:
    df (pandas.DataFrame): The input dataframe.
    column_name (str): The name of the column to remove outliers from.
    window_size (int): The size of the sliding window.
    sigma (int): The number of standard deviations to consider as an outlier.

    Returns:
    pandas.DataFrame: The input dataframe with outliers removed from the specified column.
    """
    # Copy the input dataframe to avoid modifying the original
    df = df.copy()

    # Define the sliding window
    window = df.rolling(window_size)

    # Calculate the mean and standard deviation for each window
    window_mean = window.mean()
    window_std = window.std()

    # Calculate the lower and upper bounds for outlier detection
    lower_bound = window_mean - sigma * window_std
    upper_bound = window_mean + sigma * window_std

    # Replace outliers with NaN
    outliers = (df < lower_bound) | (df > upper_bound)
    # df.loc[(df[column_name] < lower_bound) | (df[column_name] > upper_bound), column_name] = np.nan

    # # Forward-fill the NaN values to replace outliers with the previous valid value
    # df[column_name] = df[column_name].ffill()

    # # Backward-fill the NaN values to replace outliers at the start of the series with the next valid value
    # df[column_name] = df[column_name].bfill()

    return outliers

def outliers_IQR(df, column):
    # Calculate the interquartile range (IQR)
    Q1 = df[column].quantile(0.001)
    Q3 = df[column].quantile(0.99)
    IQR = Q3 - Q1

    # Define the lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find the outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Replace the outliers with NaN values
    df.loc[outliers.index, column] = np.nan

    return df, outliers
    
def plots(df, plot_dir):
    for col in df.columns[1:]:
        if 'WR10' in col:
            continue

        fig, axs = plt.subplots(ncols=3, figsize=(15, 5))  # Create a subplot with three axes for each plot

        axs[0].plot(df[col])  # Plot the original data
        axs[0].set_title(col)

        df_clean, outliers = outliers_IQR(df, col)  # Apply the outlier removal function

        axs[1].scatter(df.index, df[col], color='blue', marker='+')  # Plot the original data with outliers in red
        axs[1].scatter(outliers.index, outliers[col], color='red', marker='+')
        axs[1].set_title(f'{col} (outliers)')
        
        axs[2].plot(df_clean[col])  # Plot the cleaned data
        axs[2].set_title(f'{col} (clean)') 

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.savefig(f'{plot_dir}/{col}_outliers.png')
        plt.tight_layout()
        # plt.show()
    return df_clean

directory = 'Data/Filtered_data'

for file in os.listdir(directory):
    file_name = file.split('.')[0]
    print(file_name)
    # Construct the full path to the directory
    file_path = os.path.join(directory, file)
    print(file_path)
    df = pd.read_csv(file_path)
    plot_dir = f'Figures/{file_name}'
    df_clean = plots(df, plot_dir)
    df_clean.to_csv(f'Data/Removed_outliers/{file_name}.csv', index=False)
    





