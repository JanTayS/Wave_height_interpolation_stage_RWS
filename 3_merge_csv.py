import os
import pandas as pd

# Set the path to the directory containing the CSV files
main_directory = 'Data/Merged_data'
subdirectories = os.listdir(main_directory)
output_directory = 'Data/Merge_csv'

for subdir in subdirectories:
    print(subdir)
    directory = os.path.join(main_directory, subdir)

    output_name = directory.split('\\')[-1]
    # output_name = f'{directory}/{output_name}.csv'
    output_path = f'{output_directory}/{output_name}.csv'
    print(output_path)
    # Initialize an empty list to store the data frames
    df_list = []
    
    # Loop through the CSV files in the directory
    for filename in os.listdir(directory):
        if filename == f'{output_name}.csv':
            continue
        if filename.endswith('.csv'):
            # Load the CSV file into a data frame
            df = pd.read_csv(os.path.join(directory, filename))
            
            # Rename the second column to a unique name based on the filename
            df = df.rename(columns={df.columns[1]: os.path.splitext(filename)[0]})
            
            # Add the data frame to the list
            df_list.append(df)

    merged_df = pd.DataFrame()
    for dataframe in df_list:
        for column in dataframe:
            if column not in merged_df.columns:
                merged_df[column] = dataframe[column]

    # Save the merged data frame to a new CSV file
    merged_df.to_csv(output_path, index=False)