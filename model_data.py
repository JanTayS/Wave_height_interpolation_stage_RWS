import pandas as pd
import numpy as np
from get_variables import get_distance_xy
from get_variables import get_locations_dict
from get_variables import select_dataset
import os

class model_data():
    def __init__(self,df,split_wind=True):
        self.raw_dataset = df.copy()
        self.variables = ['datetime','Hm0','WS10','WR10','PQFF10']
        self.dataset = select_dataset(self.raw_dataset,self.variables)
        self.locations = get_locations_dict(self.dataset)
        self.target_variable = 'Hm0'
        if split_wind:
            self.get_wind_xy()

    def get_wind_xy(self, delete_non_directional_wind=True):
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

    def add_past_data(self, past_data_count=6):
        for column in self.dataset.columns:
            for past_data in range(past_data_count):
                self.dataset[f'{column}_-{past_data}'] = self.dataset[column].shift(-past_data)
    

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
        feature_dataset[location_target_variable] = feature_dataset[location_target_variable].median()
        feature_dataset = feature_dataset.dropna(subset=['target'])
        return feature_dataset
    
    def drop_impute(self, dataset):
        input_variables = [input_variable for input_variable in dataset.columns if input_variable != 'target']
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
            target_dataset = self.create_target(location_target_variable, location_dataset)
            if stacked_data.empty:
                stacked_data = target_dataset
            else:
                stacked_data = pd.concat([stacked_data,target_dataset], axis=0)
        stacked_data.reset_index(drop=True)
        stacked_data = self.drop_impute(stacked_data)
        return stacked_data

    def creat_dataset(self, location_target_variable):
        model_dataset = self.get_location_data(location_target_variable)
        model_dataset = self.drop_impute(model_dataset)
        model_dataset = self.create_target(location_target_variable,model_dataset)        
        model_dataset.reset_index(drop=True)
        return model_dataset
    
    def create_dataset_loop(self):
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
            if os.path.exists(f'model_datasets/{dataset_name}.csv'):
                i = 1
                while os.path.exists(f'model_datasets/{dataset_name}_{i}.csv'):
                    i += 1
                dataset_name = f"{dataset_name}_{i}"
            model_dataset.to_csv(f'model_datasets/{dataset_name}.csv', index=False)

if __name__ == '__main__':
    df = pd.read_csv('final_data.csv')
    data_modelling = model_data(df)
    model_dataset = data_modelling.stack_location_data()
    model_dataset.to_csv('model_datasets/model_dataset2.csv',index=False)    

    # data_modelling.create_dataset_loop()
    


    # location = 'Hm0_D151'
    # model_dataset = data_modelling.creat_dataset(location)
    # dataset_name = f'model_dataset_{location}'
    # if os.path.exists(f'model_datasets/{dataset_name}.csv'):
    #     i = 1
    #     while os.path.exists(f'model_datasets/{dataset_name}_{i}.csv'):
    #         i += 1
    #     dataset_name = f"{dataset_name}_{i}"
    # model_dataset.to_csv(f'model_datasets/{dataset_name}.csv',index=False)
    