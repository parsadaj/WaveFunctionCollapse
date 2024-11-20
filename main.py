# import os
# import glob

import numpy as np
import rasterio

from WFC import WaveFunctionCollapse
from utils import save_state
from functions import height_to_slopes, slopes_to_height

import os

# reading data
# sample_data_path = "data/N26E057.hgt"
# sample_data_path = "data/N27E060.hgt"
# sample_data_path = "data/N28E057.hgt"


for sample_data_path in ["data/N28E057.hgt", "data/N29E058.hgt", "data/N30E054.hgt", "data/N31E051.hgt", "data/N35E047.hgt"]:
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
    wfc_terrain.match_patterns()

    dirname = os.path.splitext(os.path.basename(sample_data_path))[0]
    os.makedirs(f"./results/{dirname}", exist_ok=True)
    save_state(wfc_terrain, f'./results/{dirname}/wfc_state.pkl')
    # save_state(wfc_terrain.pattern_frequencies, 'wfc_patterns.pkl')
    # save_state(wfc_terrain.adjacency_rules, 'wfc_adjacency.pkl')
    # save_state(wfc_terrain.pattern_to_number, 'wfc_pattern2num.pkl')

    # running WFC
    n_out = 0

    while n_out < 2:
        output_image = wfc_terrain.run()
        save_state(output_image, f'./results/{dirname}/wfc_out.pkl')
        n_out += 1

    print(f"done with {dirname}...")
    print(f"proceding to next file...")
    print("------------------------")
    print()    