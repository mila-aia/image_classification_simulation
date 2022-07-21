import typing
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch
from sklearn import metrics

def get_clustering_metrics(labels_true: np.array, labels_predicted: np.array)->dict:
    """
    Computes the clustering metrics.

    Parameters
    ----------
    labels_true : np.array of shape (n_samples,)
        The true labels.
    labels_predicted : np.array, shape (n_samples)
        The predicted labels.
    """
    # https://scikit-learn.org/stable/modules/clustering.html#rand-score
    m = {
        "rand_score": metrics.adjusted_rand_score(labels_true, labels_predicted),
        "adjusted_rand_score": metrics.adjusted_rand_score(labels_true, labels_predicted),
        "mutual_info_score": metrics.mutual_info_score(labels_true, labels_predicted),
    }
    return m


def show_grid_images(
    images: typing.List[torch.Tensor],
    label: list,
    height: int = 8,
    width: int = 8,
    save_path: str = None,
):
    """Show a grid of images.

    Parameters
    ----------
    images : _type_
        list of images
    label : _type_
        _description_
    height : int, optional
        image height, by default 8
    width : int, optional
        image width, by default 8
    save_path : str, optional
        path to save image, by default None
    """
    grid = make_grid(images)
    plt.figure(figsize=(height, width))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("cluster {}".format(label))
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)


def dataloader_to_images(
    dataloader: DataLoader,
) -> typing.Union[typing.List[torch.Tensor], list]:
    """Convert a dataloader to a list of images.

    Parameters
    ----------
    dataloader : DataLoader
        dataloader to convert to a list of images

    Returns
    -------
    typing.Union[typing.List[torch.Tensor], list]
        list of images and labels
    """
    images, labels = [], []
    for batch_images, batch_labels in dataloader:
        images.extend(batch_images.detach().cpu())
        labels.extend(batch_labels.detach().cpu().tolist())
    return images, labels


def show_images_in_clusters(
    images: typing.List[torch.Tensor], predicted_clusters: np.ndarray
):
    """Show images in clusters.

    Parameters
    ----------
    images : typing.List[torch.Tensor]
        list of images
    predicted_clusters : np.ndarray
        predicted clusters indices
    """
    num_clusters = np.unique(predicted_clusters)
    for num in num_clusters:
        print(
            "cluster {} has {} images".format(
                num, np.sum(predicted_clusters == num)
            )
        )
        images_in_cluster = np.array(images)[predicted_clusters == num]
        show_grid_images(images_in_cluster.tolist(), num)


def get_clustering_alg(hparams: dict):
    """Initialize the class.

    Parameters
    ----------
    hparams : dict
        hyperparameters
    """
    if hparams["clustering_alg"] == "MiniBatchKMeans":
        clustering_alg = MiniBatchKMeans(
            n_clusters=hparams["num_clusters"],
            random_state=hparams["random_state"],
            batch_size=hparams["clustering_batch_size"],
            reassignment_ratio=hparams["reassignment_ratio"],
            verbose=1,
        )
    if hparams["clustering_alg"] == "BIRCH":
        clustering_alg = Birch(
            # threshold=12.5,
            n_clusters=hparams["num_clusters"]
        )
    return clustering_alg


def get_inertia(clustering_alg):
    """Get the inertia of the clustering algorithm."""
    return clustering_alg.inertia_


def visualize(dataloader: DataLoader, predicted_clusters):
    """Visualize the clusters.

    Parameters
    ----------
    dataloader : DataLoader
        dataloader can be train, validation or test dataloader
    """
    images, labels = dataloader_to_images(dataloader)
    # predicted_clusters = self.predict(dataloader)
    show_images_in_clusters(images, predicted_clusters)
    return predicted_clusters
