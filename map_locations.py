import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import folium

directory = 'location_distance2.csv'

df = pd.read_csv(directory)


# Create a map centered at a specific location
m = folium.Map(location=[52.960539, 3.578825], zoom_start=10)

# Iterate through your DataFrame to add markers for each location
for index, row in df.iterrows():
    latitude = row['CoordinateNB']
    longitude = row['CoordinateOL']
    location_name = row['LocationCode']
    
    # Create a marker for each location
    folium.Marker([latitude, longitude], popup=location_name).add_to(m)

# Display the map
m.save("map.html")