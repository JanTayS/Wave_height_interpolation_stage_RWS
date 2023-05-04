import os
import pandas as pd

# Set the path to the directory containing the CSV files
main_directory = 'Data/Merged_data'
subdirectories = os.listdir(main_directory)
output_directory = 'Data/Final_data'

for subdir in subdirectories:
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
            
            # Convert the first column to datetime format
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            
            # Rename the second column to a unique name based on the filename
            df = df.rename(columns={df.columns[1]: os.path.splitext(filename)[0]})
            
            # Add the data frame to the list
            df_list.append(df)

    # Merge the data frames on the datetime column
    merged_df = pd.concat(df_list, axis=1)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    merged_df = merged_df.drop_duplicates(subset=[merged_df.columns[0]], keep='first')

    # Save the merged data frame to a new CSV file
    merged_df.to_csv(output_path, index=False)