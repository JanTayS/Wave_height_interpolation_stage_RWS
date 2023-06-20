import pandas as pd
import numpy as np
from get_variables import get_distance_xy
from get_variables import get_locations_dict
from get_variables import select_dataset
import os

class model_data():
    def __init__(self,df,split_wind=True, add_past_hours=True):
        self.raw_dataset = df.copy()
        self.variables = ['datetime','Hm0','WS10','WR10','PQFF10']
        self.dataset = select_dataset(self.raw_dataset,self.variables)
        self.locations = get_locations_dict(self.dataset)
        self.target_variable = 'Hm0'
        if split_wind:
            self.get_wind_xy()
        if add_past_hours:
            self.calculate_hourly_averages()

    def get_wind_xy(self, delete_non_directional_wind=False):
        # Identify the columns for wind speed and wind direction
        wind_speed_columns = []
        wind_direction_columns = []

        for column in self.dataset.columns:
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
                wind_direction_rad = np.radians(self.dataset[wind_direction_column])
                wind_direction_x = np.cos(wind_direction_rad)
                wind_direction_y = np.sin(wind_direction_rad)
                combined_column_x = f'wind_x_{location}'
                combined_column_y = f'wind_y_{location}'
                self.dataset[combined_column_x] = wind_direction_x * self.dataset[wind_speed_column]
                self.dataset[combined_column_y] = wind_direction_y * self.dataset[wind_speed_column]

        columns_to_delete = []
        for column in self.dataset.columns:
            if 'WR10' in column:
                columns_to_delete.append(column)
            if delete_non_directional_wind:
                if 'WS10' in column:
                    columns_to_delete.append(column)
        self.dataset = self.dataset.drop(columns=columns_to_delete)

    def calculate_hourly_averages(self, num_hours=6):
        for column in self.dataset.columns:
            if 'Hm0' in column:
                col_name1 = f'{column}_hour_avg_1'
                window_size = 6
                self.dataset[col_name1] = self.dataset[column].rolling(window=window_size).mean()
                for i in range(1, num_hours):
                    col_name = f'{column}_hour_avg_{i}'
                    self.dataset[col_name] = self.dataset[col_name1].shift(window_size*i)

    def get_location_data(self, location_target_variable):
        location_dataset = self.dataset.copy()
        target_location = location_target_variable.split('_')[1]
        for location in self.locations.keys():
            distance_x, distance_y = get_distance_xy(location,target_location)
            location_dataset[f'distance_x_{location}'] = distance_x
            location_dataset[f'distance_y_{location}'] = distance_y
        return location_dataset
    
    def create_target(self, location_target_variable, feature_dataset):
        feature_dataset['target'] = feature_dataset[location_target_variable]
        variable, location = location_target_variable.split('_')
        for column in feature_dataset.columns:
            if variable in column and location in column:
                feature_dataset[column] = feature_dataset[column].median()
        # feature_dataset[location_target_variable] = feature_dataset[location_target_variable].median()
        feature_dataset = feature_dataset.dropna(subset=['target'])
        return feature_dataset
    
    def drop_impute(self, dataset, location_target_variable):
        input_variables = [input_variable for input_variable in dataset.columns if input_variable != location_target_variable]
        dataset[input_variables] = dataset[input_variables].fillna(dataset[input_variables].median())
        return dataset

    def stack_location_data(self):
        stacked_data = pd.DataFrame()
        for location in self.locations.keys():
            print(location)
            location_target_variable = None
            for location_variable in self.locations[location]:
                if self.target_variable in location_variable:
                    location_target_variable = location_variable
            if location_target_variable == None:
                continue
            location_dataset = self.get_location_data(location_target_variable)
            model_dataset = self.drop_impute(location_dataset, location_target_variable)
            target_dataset = self.create_target(location_target_variable, model_dataset)
            if stacked_data.empty:
                stacked_data = target_dataset
            else:
                stacked_data = pd.concat([stacked_data,target_dataset], axis=0)
        
        stacked_data.reset_index(drop=True)
        return stacked_data

    def creat_dataset(self, location_target_variable):
        model_dataset = self.get_location_data(location_target_variable)
        model_dataset = self.drop_impute(model_dataset, location_target_variable)
        model_dataset = self.create_target(location_target_variable,model_dataset)        
        model_dataset.reset_index(drop=True)
        return model_dataset
    
    def create_dataset_loop(self):
        version = 0
        directory = f'model_datasets/version_{version}'
        while os.path.exists(directory):
            version += 1
            directory = f'model_datasets/version_{version}'

        os.makedirs(directory)

        stacked_data = pd.DataFrame()
        stacked_training_data = pd.DataFrame()
        stacked_test_data = pd.DataFrame()
        for location in self.locations.keys():
            print(location)
            location_target_variable = None
            for location_variable in self.locations[location]:
                if self.target_variable in location_variable:
                    location_target_variable = location_variable
            if location_target_variable == None:
                continue
            model_dataset = self.creat_dataset(location_target_variable)
            dataset_name = f'model_dataset_{location_target_variable}'
            
            model_dataset.to_csv(f'{directory}/{dataset_name}.csv', index=False)

            model_dataset['datetime'] = pd.to_datetime(df['datetime'])
            model_dataset = model_dataset.sort_values('datetime')

            train_start_date = '2017-01-01'
            train_end_date = '2021-12-31'
            test_start_date = '2022-01-01'
            test_end_date = '2022-12-31'

            train_data = model_dataset[(model_dataset['datetime'] >= train_start_date) & (model_dataset['datetime'] <= train_end_date)]
            test_data = model_dataset[(model_dataset['datetime'] >= test_start_date) & (model_dataset['datetime'] <= test_end_date)]

            train_data = train_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)
            
            if stacked_data.empty:
                stacked_data = model_dataset
                stacked_training_data = train_data
                stacked_test_data = test_data
            else:
                stacked_data = pd.concat([stacked_data,model_dataset], axis=0)
                stacked_training_data = pd.concat([stacked_training_data,train_data], axis=0)
                stacked_test_data = pd.concat([stacked_test_data,test_data], axis=0)
            
        print('saving complete dataset')
        stacked_data.to_csv(f'{directory}/model_dataset_all.csv', index=False)
        print('saving training dataset')
        stacked_training_data.to_csv(f'{directory}/model_dataset_training.csv', index=False)
        print('saving test dataset')
        stacked_test_data.to_csv(f'{directory}/model_dataset_test.csv', index=False)

if __name__ == '__main__':
    df = pd.read_csv('final_data.csv')
    data_modelling = model_data(df,add_past_hours=False)
    # model_dataset = data_modelling.stack_location_data()
    # model_dataset.to_csv('model_datasets/model_dataset.csv',index=False)    

    data_modelling.create_dataset_loop()
    


    # location = 'Hm0_K141'
    # model_dataset = data_modelling.creat_dataset(location)
    # dataset_name = f'model_dataset_{location}'
    # if os.path.exists(f'model_datasets/{dataset_name}.csv'):
    #     i = 1
    #     while os.path.exists(f'model_datasets/{dataset_name}_{i}.csv'):
    #         i += 1
    #     dataset_name = f"{dataset_name}_{i}"
    # model_dataset.to_csv(f'model_datasets/{dataset_name}.csv',index=False)
    