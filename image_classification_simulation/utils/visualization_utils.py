import matplotlib.pyplot as plt
import numpy as np


def show_grid_images(
    images: list,
    # labels=query_result["labels"],
    num_rows=5,
    num_cols=5,
    save_path=None,
):
    """Shows a list of images in a grid.

    Parameters
    ----------
    images : list
        List of images to show.
    num_rows : int, optional
        Number of rows in the grid. The default is 5.
    num_cols : int, optional
        Number of columns in the grid. The default is 5.
    save_path : str, optional
        Path to save the grid image. The default is None.
    """
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    for i in range(num_rows):
        for j in range(num_cols):
            try:
                img_path = images[i * num_cols + j]
                img = plt.imread(img_path)
            except IndexError:
                img = np.ones((300, 300, 3))
            ax[i, j].imshow(img)
            # ax[i, j].set_title(labels[i * num_cols + j])
            ax[i, j].axis("off")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    return fig, ax
