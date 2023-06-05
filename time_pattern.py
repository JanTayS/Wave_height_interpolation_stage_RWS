import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('merged_file.csv')
variable = 'WS10'

# Convert datetime column to datetime data type
df['datetime'] = pd.to_datetime(df['datetime'])

columns_to_analyze = []
# Define the list of columns to analyze
for column in df.columns:
    if variable in column:
        columns_to_analyze.append(column)

# Create subplots for time of year analysis
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

for i, column in enumerate(columns_to_analyze):
    # Extract time of year (month) from datetime column
    df['time_of_year'] = df['datetime'].dt.month

    # Time of Year Analysis
    time_of_year_stats = df.groupby('time_of_year')[column].mean()
    ax1.plot(time_of_year_stats.index, time_of_year_stats.values, marker='o', label=column)

ax1.set_xlabel('Time of Year (Month)')
ax1.set_ylabel('Median of {}'.format(column))
ax1.set_title('Median of {} by Time of Year'.format(column))
ax1.legend()

# Create subplots for time of day analysis
fig, ax3 = plt.subplots(figsize=(10, 6))
ax4 = ax3.twinx()

for i, column in enumerate(columns_to_analyze):
    # Extract time of day (hour) from datetime column
    df['time_of_day'] = df['datetime'].dt.hour

    # Time of Day Analysis
    time_of_day_stats = df.groupby('time_of_day')[column].mean()
    ax3.plot(time_of_day_stats.index, time_of_day_stats.values, marker='o', label=column)

ax3.set_xlabel('Time of Day (Hour)')
ax3.set_ylabel('Mean of {}'.format(column))
ax3.set_title('Mean of {} by Time of Day'.format(column))
ax3.legend()

# Adjust the layout and show the plots
plt.tight_layout()
plt.show()
