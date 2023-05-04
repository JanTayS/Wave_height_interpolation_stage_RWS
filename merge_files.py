import os
import pandas as pd

main_directory = 'Data/Unpacked_data'
output_directory = 'Data/Merged_data'
subdirectories = os.listdir(main_directory)

for subdir in subdirectories:
    directory = os.path.join(main_directory, subdir)

    sub_subdirectories = os.listdir(directory)
    for sub_subdir in sub_subdirectories:
        subdirectory = os.path.join(directory, sub_subdir)  
        output_file = os.path.join(output_directory, subdir, sub_subdir + '.csv')

        # Get a list of all the files in the directory
        file_list = os.listdir(subdirectory)

        # Sort the list of files by date
        file_list.sort()
        
        # Initialize an empty DataFrame to hold the data
        df_all = pd.DataFrame()

        # Loop through each file and append the data to the DataFrame
        for filename in file_list:
            filepath = os.path.join(subdirectory, filename)
            df = pd.read_csv(filepath, delimiter='>', header=None, names=['datetime', 'value'])
            df_all = pd.concat([df_all, df])

        # Write the merged DataFrame to a single file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_all.to_csv(output_file, index=False)