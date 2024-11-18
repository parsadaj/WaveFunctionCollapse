import matplotlib.pyplot as plt
import numpy as np

def visualize_terrain(terrain, cmap='terrain'):
    """
    Visualize the terrain heightmap using matplotlib.
    
    Args:
        terrain (numpy.ndarray): Terrain heightmap.
        cmap (str, optional): Colormap to use for visualization. Default is 'terrain'.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(terrain, cmap=cmap) #, vmin=0, vmax=1)
    # ax.set_xticks([])
    # ax.set_yticks([])
    fig.colorbar(im, ax=ax)
    plt.show()



def visualize_3d_terrain(terrain, cmap='terrain', elev=45, azim=45, x_scale=30, y_scale=30, z_scale=1):
    """
    Visualize the terrain heightmap in 3D using matplotlib.
    
    Args:
        terrain (numpy.ndarray): Terrain heightmap.
        cmap (str, optional): Colormap to use for visualization. Default is 'terrain'.
        elev (float, optional): Elevation angle for the 3D view. Default is 45 degrees.
        azim (float, optional): Azimuth angle for the 3D view. Default is 45 degrees.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = -np.arange(terrain.shape[0]) * x_scale
    y = np.arange(terrain.shape[1]) * y_scale
    X, Y = np.meshgrid(x, y)
    Z = terrain * z_scale
    ax.plot_surface(X, Y, Z, cmap=cmap, rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=elev, azim=azim)
    plt.show()
    
    
def read_hgt(filename):
    # .hgt files contain 16-bit signed integer elevation data
    with open(filename, 'rb') as file:
        # Determine the size of the grid (usually 3601x3601 or 1201x1201)
        file_size = file.seek(0, 2)  # Go to end of file to check size
        file.seek(0)  # Reset to the beginning
        
        if file_size == 3601 * 3601 * 2:  # 3601x3601 grid
            shape = (3601, 3601)
        elif file_size == 1201 * 1201 * 2:  # 1201x1201 grid
            shape = (1201, 1201)
        else:
            raise ValueError("Unknown .hgt file size: {}".format(file_size))

        # Read the data as 16-bit signed integers
        elevation_data = np.fromfile(file, np.int16).reshape(shape)
        return elevation_data
