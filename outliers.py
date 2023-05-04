import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

# Load the data from a pandas dataframe
df = pd.read_csv('Data/Final_data/AWG1_filtered.csv')

column = 'Hm0'

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
    Q1 = df[column].quantile(0.01)
    Q3 = df[column].quantile(0.99)
    IQR = Q3 - Q1

    # Define the lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find the outliers
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_clean, outliers
    

def plot_outliers(df, column, outliers):
    # Plot the data with outliers highlighted in red
    plt.scatter(df.index, df[column], color='red', marker='+')
    plt.scatter(outliers.index, outliers, color='blue', marker='+')
    plt.show()

# Plot each column in a separate subplot
# fig, axs = plt.subplots(nrows=len(df.columns), figsize=(8, 10))
# for i, col in enumerate(df.columns):
#     axs[i].plot(df[col])
#     axs[i].set_title(col)
# plt.tight_layout()
# plt.show()


for col in df.columns[1:]:
    if col == 'WR10':
        continue
    plt.figure()  # Create a new figure for each plot
    plt.plot(df[col])
    plt.title(col)  # Use the column name as the title of the plot
    plt.show()
    
    df_clean, outliers = outliers_IQR(df, col)
    plot_outliers(df,col,outliers)
    

# df_clean, outliers = outliers_IQR(df, column)
# plot_outliers(df,column,outliers)


# plt.plot(df_clean[column])
# plt.show()

# print(df_clean.describe())

# df_clean.to_csv('Data/Test_data/test.csv', index=False)
