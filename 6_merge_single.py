import pandas as pd
import numpy as np
import os
from datetime import datetime

directory = 'Data/Removed_outliers'

merged_df = pd.DataFrame()

variable = 'WS10'

file_list = os.listdir(directory)


for file in file_list:
    # Construct the full path to the directory
    file_path = os.path.join(directory, file)
    file_name = file.split('.')[0]
    df = pd.read_csv(file_path)
    print(file_name)
    
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%y %H:%M:%S ')
    for column in df.columns:
        if column == 'datetime':
            continue
        df = df.rename(columns= {column: f'{column}_{file_name}'})
    df = df.drop_duplicates('datetime')
    
    if not merged_df.empty:
        merged_df = pd.merge(df,merged_df, how='outer', on='datetime')
    else:
        merged_df = df
      

merged_df = merged_df.reset_index(drop=True)
merged_df.to_csv('final_data.csv', index=False)