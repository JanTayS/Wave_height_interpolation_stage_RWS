import pandas as pd

df = pd.read_csv('merged_file.csv')

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')

train_start_date = '2017-01-01'
train_end_date = '2021-12-31'
test_start_date = '2022-01-01'
test_end_date = '2022-12-31'

train_data = df[(df['datetime'] >= train_start_date) & (df['datetime'] <= train_end_date)]
test_data = df[(df['datetime'] >= test_start_date) & (df['datetime'] <= test_end_date)]

train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

print(train_data)
print(test_data)

train_data.to_csv('train.csv')
test_data.to_csv('test.csv')