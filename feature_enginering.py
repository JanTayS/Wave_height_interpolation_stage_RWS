import numpy as np
import pandas as pd


df = pd.read_csv('final_data.csv')

# Identify the columns for wind speed and wind direction
wind_speed_columns = []
wind_direction_columns = []
locations_with_both = []

for column in df.columns:
    if column.split('_')[0].endswith('WS10'):
        location = column.split('_')[1]
        wind_speed_columns.append((column, location))
    elif column.split('_')[0].endswith('WR10'):
        location = column.split('_')[1]
        wind_direction_columns.append((column, location))

# Combine wind direction components for each location
for _, location in wind_speed_columns:
    if (wind_speed_column := next((col for col, loc in wind_speed_columns if loc == location), None)) and \
            (wind_direction_column := next((col for col, loc in wind_direction_columns if loc == location), None)):
        wind_direction_rad = np.radians(df[wind_direction_column])
        wind_direction_x = np.cos(wind_direction_rad)
        wind_direction_y = np.sin(wind_direction_rad)
        combined_column_x = f'Wind_x_{location}'
        combined_column_y = f'Wind_y_{location}'
        df[combined_column_x] = wind_direction_x * df[wind_speed_column]
        df[combined_column_y] = wind_direction_y * df[wind_speed_column]

# Print the updated DataFrame
print(df)

df.to_csv('wind.csv', index=False)
