import pandas as pd
from math import sqrt

def get_distance_xy(origin, destination):
    df = pd.read_csv('location_distance.csv', index_col=0)
    for location_code in df.index:
        if origin in location_code:
            origin = location_code
        if destination in location_code:
            destination = location_code
    distance_x = df.loc[origin][f'{destination}_x']
    distance_y = df.loc[origin][f'{destination}_y']
    return distance_x, distance_y

def get_location_variables(df, location, variables=['Hm0']):
    location_variables = pd.DataFrame()
    for variable in variables:
        for column in df.columns:
            if location in column and variable in column:
                location_variables[column] = df[column]
    return location_variables

def get_location_columns(df, locations, input_variables=['Hm0','WS10','WR10']):
    input_columns = []
    for variable in input_variables:
        for location in locations:
            for column in df.columns:
                if location in column and variable in column:
                 input_columns.append(column)
    return input_columns

def get_locations(df):
    locations = set()
    for column in df.columns:
        if '_' in column and 'Hm0' in column:
            location = column.split('_')[1]
            locations.add(location)
    return sorted(list(locations))

def get_locations_dict(df):
    locations = get_locations(df)
    locations_dict = {}
    for location in locations:
        location_variables = []
        for column in df.columns:
            if location in column:
                location_variables.append(column)
        locations_dict[location] = location_variables
    return locations_dict

def select_dataset(df, variables):
    columns = []
    for variable in variables:
        for column in df.columns:
            if variable in column:
                columns.append(column)
    selection_dataset = df[columns]
    return selection_dataset

def remove_nan(df, min_rows=0):
    clean_df = df.copy()
    while clean_df.dropna().shape[0] <= min_rows:
        missing_values = clean_df.isna().sum()
        max_missing = missing_values.idxmax()
        clean_df = clean_df.drop(max_missing, axis=1)
    clean_df = clean_df.dropna()
    return clean_df

def get_nearby_locations(df, origin, amount=5, keep_duplicate_locations=False):
    distance_locations = {}
    locations = get_locations(df)
    for location in locations:
        distance_x, distance_y = get_distance_xy(origin,location)
        euclidian_distance = sqrt(distance_x**2 + distance_y**2)
        distance_locations[location] = euclidian_distance
    
    distance_locations = {key: value for key, value in distance_locations.items() if value != 0} # drop location where distance is 0
    sorted_distance = sorted(distance_locations.items(), key=lambda x: x[1])
    nearby_locations = [key for key, _ in sorted_distance[:amount]]
    return nearby_locations

def get_regression_df(df,target_location='K141',input_variables=['Hm0','WS10','WR10']):
    df_regression = df.copy()
    nearby_locations = get_nearby_locations(df_regression, target_location)
    input_columns = get_location_columns(df_regression, nearby_locations, input_variables)
    all_columns = input_columns + [f'Hm0_{target_location}']
    df_regression = df_regression[all_columns]
    df_regression = remove_nan(df_regression)
    return df_regression

def get_regression_df2(df,target_location='K141',input_variables=['Hm0','WS10','WR10']):
    df_regression = df.copy()
    nearby_locations = get_nearby_locations(df_regression, target_location)
    input_columns = get_location_columns(df_regression, nearby_locations, input_variables)
    all_columns = input_columns + [f'Hm0_{target_location}']
    df_regression = df_regression[all_columns]
    return df_regression
    


if __name__ == "__main__":
    df_main = pd.read_csv('final_data.csv')
    df = df_main.copy()
    target_location = 'K141'
    nearby_locations = get_nearby_locations(df, target_location)
    input_columns = get_location_columns(df, nearby_locations)
    df = df[input_columns]
    print(df.shape)
    df = remove_nan(df)
    print(df.shape)
    print(df.isna().sum())
