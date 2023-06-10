import folium
from folium.plugins import HeatMap

# Create a map centered around the North Sea
north_sea_map = folium.Map(location=[55.0, 2.0], zoom_start=6)

# Define the locations and wave heights
locations = {
    'Location A': {'lat': 54.0, 'lon': 1.0, 'wave_height': 1.5},
    'Location B': {'lat': 55.5, 'lon': 3.0, 'wave_height': 2.3},
    'Location C': {'lat': 56.5, 'lon': 2.5, 'wave_height': 0.8},
    # Add more locations and their respective wave heights
}

# Prepare data for the heatmap
heat_data = [[data['lat'], data['lon'], data['wave_height']] for data in locations.values()]

# Add the heatmap layer to the map
HeatMap(heat_data).add_to(north_sea_map)

# Save the map as an HTML file
north_sea_map.save('north_sea_map.html')