import pandas as pd

df = pd.read_csv('final_data.csv')

def calculate_hourly_averages(df, num_hours):
        time_dataset = df.copy()
        for column in time_dataset.columns:
            if 'Hm0' in column:
                col_name1 = f'{column}_avg_1_hour_ago'
                window_size = 6
                time_dataset[col_name1] = time_dataset[column].rolling(window=window_size).mean()
                for i in range(1, num_hours):
                    col_name = f'{column}_avg_{i}_hour'
                    time_dataset[col_name] = time_dataset[col_name1].shift(window_size*i)
        return time_dataset

time_dataset = calculate_hourly_averages(df,6)
time_dataset.to_csv('time_dataset_test.csv')