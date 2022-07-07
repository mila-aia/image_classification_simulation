import typing
import numpy as np
import matplotlib.pyplot as plt
from torch import device, cuda
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, torch
from image_classification_simulation.models.model_loader import load_model
from image_classification_simulation.data.office31_loader import Office31Loader
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch


def show_grid_images(
    images: typing.List[torch.Tensor],
    label: list,
    height: int = 8,
    width: int = 8,
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
    """
    grid = make_grid(images)
    plt.figure(figsize=(height, width))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("cluster {}".format(label))
    plt.show()


def dataloader_to_images(
    dataloader: DataLoader,
) -> typing.Union[typing.List[torch.Tensor], list]:
    images, labels = [], []
    for batch_images, batch_labels in dataloader:
        images.extend(batch_images.detach().cpu())
        labels.extend(batch_labels.detach().cpu().tolist())
    return images, labels


def show_images_in_clusters(
    images: typing.List[torch.Tensor], predicted_clusters: np.ndarray
):
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
    """The class does clustering with different algorithms"""

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


if __name__ == "__main__":
    hparams = {"num_workers": 2, "batch_size": 32, "image_size": 224}
    office_loader = Office31Loader(
        data_dir="./examples/data/domain_adaptation_images/amazon/images/",
        hyper_params=hparams,
    )
    office_loader.setup("fit")
    train_loader = office_loader.train_dataloader()
    val_loader = office_loader.val_dataloader()

    # hparams = {
    #     "clustering_alg": "MiniBatchKMeans",
    #     "loss": "CrossEntropyLoss",
    #     "pretrained": True,
    #     "num_classes": 31,
    #     "path_features_ext": "./output/best_model/model.ckpt",
    #     "architecture": "vit",
    #     "num_clusters": 31,
    #     "random_state": 0,
    #     "clustering_batch_size": 32,
    #     "reassignment_ratio": 0.05,
    # }
    # clustering = Clustering(hparams)
    # clustering.fit(train_loader)
    # clustering.visualize(val_loader)
    hparams = {
        "clustering_alg": "MiniBatchKMeans",
        "loss": "CrossEntropyLoss",
        "pretrained": True,
        "num_classes": 31,
        "path_features_ext": "./examples/resnet/output/best_model/model.ckpt",
        "architecture": "resnet",
        "num_clusters": 31,
        "random_state": 0,
        "clustering_batch_size": 32,
        "size": 256,
        "reassignment_ratio": 0.05,
    }
    clustering = Clustering(hparams)
    clustering.fit(train_loader)
    clustering.visualize(val_loader)
