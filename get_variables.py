import pandas as pd

df = pd.read_csv('final_data.csv')

def get_distance_xy(origin, destination):
    df = pd.read_csv('location_distance.csv', index_col=0)
    distance_x = df.loc[origin][f'{destination}_x']
    distance_y = df.loc[origin][f'{destination}_y']
    return distance_x, distance_y

def get_location_variables(df, location, variables=['Hm0']):
    location_variables = pd.DataFrame()
    for variable in variables:
        print(variable)
        for column in df.columns:
            if location in column and variable in column:
                location_variables[column] = df[column]
    return location_variables

print(get_location_variables(df,'A121', ['Hm0','WS10','WR10']))