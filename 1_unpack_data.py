import tarfile
import os

# Specify the path to the tgz files directory
directories = "Packed_data/JanB"

# Specify the path to the output directory where to save the regular files
data_dir = "Data"

# Loop through all directories in the directory
for directory in os.listdir(directories):
    # Construct the full path to the directory
    dir_path = os.path.join(directories, directory)
    
    # Check if the entry is a directory
    if os.path.isdir(dir_path):
        # Loop through all tgz files in the directory
        for file in os.listdir(dir_path):
            if file.endswith('.tgz'):
                # Get the year from the filename
                year = file.split('_')[0]
                name = file.split('_')[1]

                # Open the tgz file in read mode
                with tarfile.open(os.path.join(dir_path, file), 'r:gz') as tar:
                    
                    # Extract regular files to output directory
                    for member in tar.getmembers():
                        if member.type.decode() == '0':
                            # If the member is a regular file, extract it to the output directory
                            output_dir = os.path.join(data_dir, directory)
                            tar.extract(member, path=output_dir)
                            
                            # Print the name of the extracted file
                            # print(f"Extracted {member.name} to {output_dir}")