import pandas as pd
import numpy as np

# df_full = pd.read_csv('final_data.csv')
# print('data_loaded')

# location = 'L91'
# location_columns = []
# for column in df_full.columns:
#     if location in column:
#         location_columns.append(column)


# # Assuming df is your DataFrame
# df = df_full[location_columns] # Select all columns except the first one

# # Generate descriptive statistics
# desc_df = df.describe()

# # Calculate the count of NaN values per column
# nan_count = df.isnull().sum()

# # Append the count of NaN values to the descriptive statistics dataframe
# desc_df = desc_df.append(pd.Series(nan_count, name='NaN'))

# # Convert the descriptive statistics dataframe to a LaTeX table
# latex_table = desc_df.to_latex()

# print(latex_table)

test_df = pd.read_hdf('models/run_7/train.h5')
for column in test_df.columns:
    print(column)