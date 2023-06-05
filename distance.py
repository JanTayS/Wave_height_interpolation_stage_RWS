import pandas as pd
import numpy as np
import math

df = pd.read_csv('locations.csv')

def calculate_vector(origin, destination):
    # Convert latitude and longitude to radians
    lat1 = math.radians(origin[0])
    lon1 = math.radians(origin[1])
    lat2 = math.radians(destination[0])
    lon2 = math.radians(destination[1])

    # Differences in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Earth radius in kilometers
    earth_radius = 6371.0

    # X-component of the vector (vector represents the east-west displacement or distance between the origin and destination locations)
    # calculated as the difference in longitude (dlon) multiplied by the cosine of the average latitude ((lat1 + lat2) / 2.0) and then multiplied by the Earth's radius (earth_radius).
    x = dlon * math.cos((lat1 + lat2) / 2.0) * earth_radius

    # Y-component of the vector (vector represents the north-south displacement or distance between the origin and destination locations)
    # calculated as the difference in latitude (dlat) multiplied by the Earth's radius (earth_radius).
    y = dlat * earth_radius

    # Combine components to form the vector
    vector = (x, y)

    return vector

def get_lat_lon(location):
    return df.loc[df['LocationCode'] == location, 'CoordinateNB'].values[0], df.loc[df['LocationCode'] == location, 'CoordinateOL'].values[0]


for destination in df['LocationCode']:
    distances_x = []
    distances_y = []
    for origin in df['LocationCode']:
        distance_x, distance_y = calculate_vector(get_lat_lon(origin),get_lat_lon(destination))
        distances_x.append(distance_x)
        distances_y.append(distance_y)
    df[f'{destination}_x'] = distances_x
    df[f'{destination}_y'] = distances_y

df.to_csv('location_distance.csv', index=False)
print(df)

