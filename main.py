# import os
# import glob

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns

from WFC import WaveFunctionCollapse, WaveFunctionCollapseVisualizer
from utils import visualize_3d_terrain, visualize_terrain, save_state, load_state
from functions import height_to_slopes, slopes_to_height


# reading data
sample_data_path = "data/N26E057.hgt"


scale_factor = 1.0 / 8


with rasterio.open(sample_data_path) as src:
    # Define the desired smaller shape (for example, half the original size)
    new_height = int(src.height * scale_factor)
    new_width = int(src.width * scale_factor)

    # Read and resample the data to the new shape
    sample_data = src.read(
        1,  # First band
        out_shape=(new_height, new_width),
        resampling=rasterio.enums.Resampling.bilinear  # Choose the resampling method
    ).astype(float)[250:350, 200:300]
    
print(sample_data.shape)


print("data path: ", sample_data_path)

# calculating grads
grad_x, grad_y = height_to_slopes(sample_data)

# recreatinh hright from grad
hh = slopes_to_height(grad_x, grad_y)

if len(np.unique(hh-sample_data)) == 1:
    print("Recreating height succesful")
else:
    print("Recreating height failed")


# WFC Extraction
slopes = np.concatenate([grad_x[:-1, ..., np.newaxis], grad_y[..., :-1, np.newaxis]], axis=2)
wfc_terrain = WaveFunctionCollapse(slopes, (2,2,2), (20,20,2), remove_low_freq=False, low_freq=1)

save_state(wfc_terrain, 'wfc_state.pkl')

# running WFC
n_out = 0
output_image = wfc_terrain.run()
