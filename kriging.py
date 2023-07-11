import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

df_locations = pd.read_csv('locations.csv')
df_wave_height = pd.read_hdf('model_datasets/version_4/model_dataset_test.h5', key='data')




def get_lat_lon(location):
    return [df_locations.loc[df_locations['LocationCode'] == location, 'CoordinateNB'].values[0], df_locations.loc[df_locations['LocationCode'] == location, 'CoordinateOL'].values[0]]

def haversine(coord1, coord2):
    # Radius of the Earth in kilometers
    radius = 6371.0

    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = radius * c
    return distance


# wave_height_columns = [col for col in df_wave_height.columns if 'Hm0' in col]
wave_height_columns =[]
locations = []

coordinates = []
            
for col in df_wave_height.columns:
    if 'Hm0' in col:
        print(col)
        wave_height_columns.append(col)
        locations.append(col.split('_')[1])
        for location in df_locations['LocationCode']:
            if col in location:
                coordinates.append(get_lat_lon(location))
wave_height_data_full = df_wave_height[wave_height_columns].values  # Extract the underlying NumPy array
wave_height_data = wave_height_data_full[0]

coordinates = np.array(coordinates)

# Compute distances between locations using pairwise_distances function
distance_matrix = pairwise_distances(np.radians(coordinates), metric='haversine')

# Reshape wave_height_data to be 2-dimensional
wave_height_data = np.reshape(wave_height_data, (-1, len(wave_height_columns)))

# Perform Ordinary Kriging Interpolation
ok = OrdinaryKriging(
    x=coordinates[:, 1],
    y=coordinates[:, 0],
    z=wave_height_data.T,
    variogram_model="linear",
    verbose=True,
    enable_plotting=True,
)

interpolated_heights, _ = ok.execute(
    "grid",
    np.linspace(min(coordinates[:, 1]), max(coordinates[:, 1]), num=100),
    np.linspace(min(coordinates[:, 0]), max(coordinates[:, 0]), num=100),
)


# Create a grid of x and y coordinates
grid_x = np.linspace(min(coordinates[:, 1]), max(coordinates[:, 1]), num=100)
grid_y = np.linspace(min(coordinates[:, 0]), max(coordinates[:, 0]), num=100)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)

# Plot the interpolated heights as a contour plot
plt.figure(figsize=(10, 8))
plt.contourf(grid_x, grid_y, interpolated_heights, levels=20, cmap='jet')
plt.colorbar(label='Interpolated Height')
plt.scatter(coordinates[:, 1], coordinates[:, 0], c='black', marker='o')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Interpolated Wave Heights')
plt.show()