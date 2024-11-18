import numpy as np


def slopes_to_height(grad_x, grad_y):
    # # Integrate the x-gradient along the x direction
    # height_map_x = np.cumsum(grad_x, axis=1)

    # # Integrate the y-gradient along the y direction
    # height_map_y = np.cumsum(grad_y, axis=0)

    # fix, axes = plt.subplots(1, 2, figsize=(8,4))
    # # im1 = axes[0].imshow(grad_y_np)#, cmap='terrain')
    # # im2 = axes[1].imshow(grad_y)#, cmap='terrain')

    # im1 = sns.heatmap(height_map_x, ax=axes[0], annot=True)
    # im2 = sns.heatmap(height_map_y, ax=axes[1], annot=True)

    # height_map_x.shape, height_map_y.shape


    hh = np.zeros((
        grad_x.shape[0],
        grad_y.shape[1]
    ))
    
    for y in range(hh.shape[0] - 1):
        for x in range(y+1):
            if x == 0 and y < hh.shape[0]-1:
                hh[y+1, x] = hh[y,x] + grad_y[y,x]
            if x < hh.shape[1] - 1:
                hh[y,x+1] = hh[y,x] + grad_x[y,x]
            if y < hh.shape[1] - 1:
                hh[x,y+1] = hh[x,y] + grad_x[x,y]

        
    hh[-1, :] = hh[-2,:] + grad_y[-1,:]
    return hh

def height_to_slopes(heightmap):
    # # Calculate the gradient in both the x and y directions
    # grad_y_np, grad_x_np = np.gradient(heightmap)

    grad_x = heightmap[:, 1:] - heightmap[:, :-1]
    grad_y = heightmap[1:, :] - heightmap[:-1, :]
    
    return grad_x, grad_y
