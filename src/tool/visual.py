import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt

def draw_heatmap(arr: np.array, l_mask_path: str = 'data/processed/left_foot_mask.png', vmin: float = 0.0, vmax: float = 600.0, is_show: bool = True, is_export: bool = False, export_path: str = 'output.png', **kwargs):
    # load foot masks
    l_img = Image.open(l_mask_path)
    r_img = ImageOps.mirror(l_img)

    l_mask = np.array(l_img).astype(np.float64)
    r_mask = np.array(r_img).astype(np.float64)

    # detect pixels of area no.1~197 and store the corresponding indexes
    l_index = {}
    r_index = {}

    for n in range(0, 99):
        l_index[n] = np.where(l_mask == n + 1)
        r_index[n + 99] = np.where(r_mask == n + 1)

    # create left and right foot heatmap
    l_pedar = np.zeros(l_mask.shape)
    r_pedar = np.zeros(r_mask.shape)

    for idx, value in enumerate(arr):
        if idx < 99:
            # filling left foot area
            l_pedar[l_index[idx]] = value

        else:
            # filling right foot area
            r_pedar[r_index[idx]] = value

    # plot heatmap
    fig, axs = plt.subplots(1, 2)
    
    im = axs[0].imshow(l_pedar, vmin=vmin, vmax=vmax, **kwargs)
    axs[0].set_title('left')
    axs[0].axis('off')
    fig.colorbar(im, ax=axs[0])

    im = axs[1].imshow(r_pedar, vmin=vmin, vmax=vmax, **kwargs)
    axs[1].set_title('right')
    axs[1].axis('off')
    fig.colorbar(im, ax=axs[1])

    if is_export:
        plt.savefig(export_path)

    if is_show:
        plt.show()

    plt.close()
