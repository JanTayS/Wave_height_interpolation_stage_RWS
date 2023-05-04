import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

directory = 'Data/Final_data'

plot_variable = 'Hm0'

new_df = pd.DataFrame()
labels = []

for file in os.listdir(directory):
    file_name = file.split('_')[0]
    # print(file_name)

    if 'filtered' not in file:
    # if 'filtered' not in file or 'AWG1' in file or 'K131' in file:
        continue

    # Construct the full path to the directory
    file_path = os.path.join(directory, file)

    # Load the CSV file into a Pandas data frame
    df = pd.read_csv(file_path)

    # Remove the first column (contains date/time data)
    df_values = df.iloc[:, 1:]

    # Calculate the descriptive statistics for the remaining columns
    stats = df_values.describe()

    # Print the statistics
    # print(stats)
    
    # stats.to_csv(f'Statistics/summary_stats_{file_name}.csv', index=True)

    for column in df_values.columns:
        if plot_variable in column:
            # plot_values.append(df_values[column].iloc[:500])
            new_df[file_name] = df_values[column]
            labels.append(file_name)



# for values, lab in zip(plot_values,labels):
#     plt.plot(values, label=lab)

# # Add axis labels and a title
# plt.xlabel('Time')
# plt.ylabel('Significant wave height')
# plt.title(f'Plot of {plot_variable}')
# plt.legend()

# plt.savefig(f'Plots/{plot_variable}.png')
# # Display the plot
# plt.show()

# plot all columns in new_df as boxplots in a single plot
new_df.boxplot()

# set the title and axis labels
plt.xlabel('File')
plt.ylabel('Significant wave height')
plt.title(f'Boxplot of {plot_variable} for all files')

# show the plot
plt.show()

# create a histogram for each column in new_df
for col in new_df:
    plt.hist(new_df[col], bins=20, alpha=0.5, label=col)

# set the title and axis labels
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title(f'Histogram of {plot_variable} for all files')

# add a legend
plt.legend()

# show the plot
plt.show()


# loop over the variables and create a separate plot for each variable
# for values, lab in zip(plot_values,labels):
#     # create a new figure for each plot
#     fig, ax = plt.subplots()
    
#     # plot the variable
#     ax.plot(values)
    
#     # set the title and axis labels
#     ax.set_title(f'Plot of {lab}-{plot_variable}')
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Significant wave height')
    
#     # create directory if it does not exist
#     if not os.path.exists(f'Plots/{lab}'):
#                 os.makedirs(f'Plots/{lab}')

#     # save the plot to a file with a name that includes the variable name
#     plt.savefig(f'Plots/{lab}/{plot_variable}.png')
    
#     # close the figure to free up memory
#     plt.close(fig)


