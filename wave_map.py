import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import folium
import random

# Create a map centered around the North Sea
north_sea_map = folium.Map(location=[55.0, 2.0], zoom_start=6)

directory = 'location_distance2.csv'

df = pd.read_csv(directory)


# Create a map centered at a specific location
m = folium.Map(location=[52.960539, 3.578825], zoom_start=10)

locations = {}

# Iterate through your DataFrame to add markers for each location
for index, row in df.iterrows():
    latitude = row['CoordinateNB']
    longitude = row['CoordinateOL']
    location_name = row['LocationCode']
    locations[location_name] = {'lat': latitude, 'lon': longitude, 'wave_height': random.randint(0, 2000)}


# Prepare data for the heatmap
heat_data = [[data['lat'], data['lon'], data['wave_height']] for data in locations.values()]

# Add the heatmap layer to the map
HeatMap(heat_data).add_to(north_sea_map)

# Save the map as an HTML file
north_sea_map.save('north_sea_map.html')