import pandas as pd
import folium

# Read the CSV file
df = pd.read_csv('avaliable_location_distances.csv')

# Create a map centered at a specific location
m = folium.Map(location=[52.960539, 3.578825], zoom_start=10)

# Iterate through your DataFrame to add markers for each location
for index, row in df.iterrows():
    latitude = row['CoordinateNB']
    longitude = row['CoordinateOL']
    location_name = row['LocationCode']
    
    # Create a marker for each location with a custom tooltip
    tooltip = folium.Tooltip(location_name)
    folium.Marker([latitude, longitude], tooltip=tooltip).add_to(m)  # Add the marker with the tooltip to the map

# Display the map
m.save("map.html")