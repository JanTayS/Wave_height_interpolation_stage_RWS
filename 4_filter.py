import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

directory = 'Data/Merge_csv'

plot_values = []
labels = []

for file in os.listdir(directory):
    if 'filtered' in file:
        continue
    
    # Construct the full path to the directory
    file_path = os.path.join(directory, file)
    # file_path = 'Data\Final_data\BTA1.csv'

    filtered_output = file_path.split('\\')[-1]

    # Load the CSV file into a Pandas data frame
    df = pd.read_csv(file_path, low_memory=False)

    for col in df.columns[1:]:
        # split the values on the '/' character
        split_values = df[col].str.split('/', expand=True)
        
        # convert quality column to integers (if it exists)
        if len(split_values.columns) > 1:
            split_values[1] = pd.to_numeric(split_values[1], errors='coerce')
            # split_values = split_values.dropna()  # remove rows with NaN values

            # reset index to match df
            split_values = split_values.reset_index(drop=True)

            # filter out rows where quality is less than 10 / less than the max
            filtered_df = df.loc[split_values[1] >= max(split_values[1]), col]
            # filtered_df = df.loc[split_values[1] >= 10, col]
        else:
            filtered_df = df[col]

        # apply lambda function to remove quality and convert to numeric
        filtered_df = filtered_df.apply(lambda x: pd.to_numeric(x.split("/")[0], errors='coerce'))
        # update the original DataFrame with the filtered and cleaned values
        df[col] = filtered_df
    print(df)
    df.to_csv(f'Data/Filtered_data/{filtered_output}', index=False)
    print(f'{file} = done')


