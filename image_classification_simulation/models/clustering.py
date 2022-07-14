import typing
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import device, cuda
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from image_classification_simulation.models.model_loader import load_model
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch
from joblib import dump, load
from image_classification_simulation.models.model_loader import load_model


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


class Clustering:
    """The class does clustering with different algorithms."""

    def __init__(self, hparams: dict):
        """Initialize the class.

        Parameters
        ----------
        hparams : dict
            hyperparameters
        """
        self.device = device("cuda" if cuda.is_available() else "cpu")

        self.path_features_ext = hparams["path_features_ext"]

        self.feature_ext = load_model(hparams).to(self.device)
        self.feature_ext.load_from_checkpoint(
            checkpoint_path=self.path_features_ext
        )
        # have to set to eval here because batch norm
        # has no meaning for one instance
        self.feature_ext.eval()

        # last layer is usually used for task specific finetuning
        # we can remove it here, since we need features only
        layers = list(self.feature_ext.children())[-1]
        self.feature_extractor = layers

        if hparams["clustering_alg"] == "MiniBatchKMeans":
            self.clustering_alg = MiniBatchKMeans(
                n_clusters=hparams["num_clusters"],
                random_state=hparams["random_state"],
                batch_size=hparams["clustering_batch_size"],
                reassignment_ratio=hparams["reassignment_ratio"],
                verbose=1,
            )
        if hparams["clustering_alg"] == "BIRCH":
            self.clustering_alg = Birch(
                # threshold=12.5,
                n_clusters=hparams["num_clusters"]
            )

    def get_inertia(self):
        """Get the inertia of the clustering algorithm."""
        return self.clustering_alg.inertia_

    def fit(self, dataloader: DataLoader):
        """Fit the clustering algorithm.

        Parameters
        ----------
        dataloader : DataLoader
            dataloader can be train, validation or test dataloader
        """
        for batch_images, batch_labels in dataloader:
            features = self.feature_ext(batch_images.to(self.device))
            features = features.detach().cpu().numpy()
            self.clustering_alg.partial_fit(features)

    def predict(self, dataloader: DataLoader):
        """Predict the clusters for batch of images in the dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            dataloader can be train, validation or test dataloader
        """
        predicted_clusters = []
        for batch_images, batch_labels in dataloader:
            features = self.feature_ext(batch_images.to(self.device))
            features = features.detach().cpu().numpy()
            predicted_clusters.extend(self.clustering_alg.predict(features))
        return np.array(predicted_clusters)

    def predict_one_image(self, image: torch.Tensor) -> int:
        """Predict the clusters for one image.

        Parameters
        ----------
        image : torch.Tensor
            image to predict the clusters
        """
        # model needs the image to be a batch of size 1
        image = torch.unsqueeze(image, 0)
        features = self.feature_ext(image.to(self.device))
        features = features.detach().cpu().numpy()
        cluster_id = self.clustering_alg.predict(features)
        return cluster_id.item()

    def visualize(self, dataloader: DataLoader):
        """Visualize the clusters.

        Parameters
        ----------
        dataloader : DataLoader
            dataloader can be train, validation or test dataloader
        """
        images, labels = dataloader_to_images(dataloader)
        predicted_clusters = self.predict(dataloader)
        show_images_in_clusters(images, predicted_clusters)
        return predicted_clusters

    def save_model_to_file(self, path: str):
        """Save the model to a file.

        Parameters
        ----------
        path : str
            path to save the model
        """
        dump(self.clustering_alg, path)

    def load_model_from_file(self, path: str):
        """Load the model from a file.

        Parameters
        ----------
        path : str
            path to load the model
        """
        self.clustering_alg = load(path)
