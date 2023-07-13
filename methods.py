import pandas as pd
import numpy as np

data = pd.read_csv('final_data.csv')

# Remove 'datetime' from the column names
column_names = [col for col in data.columns if col != 'datetime']

# Extract variable part
variables = [name.split('_')[0].lstrip('x') for name in column_names]

# Count occurrences
variable_counts = pd.Series(variables).value_counts().reset_index()

# Create DataFrame
variables_count_df = pd.DataFrame(variable_counts)
variables_count_df.columns = ['Variable', 'Count']

variables_count_df

# Convert the DataFrame to a LaTeX table
latex_table = variables_count_df.to_latex(index=False)
print(latex_table)
