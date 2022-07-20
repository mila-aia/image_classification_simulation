from PIL import Image
import typing
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from image_classification_simulation.models.clustering_tools import (
    get_clustering_alg,
)
from image_classification_simulation.models.model_loader import load_model


def batchify(iterable: typing.Iterable, batch_size: int = 32):
    """Yield successive n-sized batches from lst.

    Parameters
    ----------
    iterable : typing.Iterable
        make an iterable from a list
    batch_size : int, optional
        batch size, by default 32

    Yields
    ------
    typing.Iterable
        batches of elements
    """
    for i in range(0, len(iterable), batch_size):
        yield iterable[i: i + batch_size]


class ImageSimilaritySearch:
    """Image similarity search class."""

    def __init__(self, hparams: dict, DataModule: DataLoader) -> None:
        """Initialize the ImageSimilaritySearch class.

        Parameters
        ----------
        hparams : dict
            Hyper-parameters for the ImageSimilaritySearch class.
        DataModule : DataLoader
            Dataset for the ImageSimilaritySearch class.
        """
        self.path_to_model = hparams["path_to_model"]
        self.path_cluster_ids = hparams["path_cluster_ids"]
        self.batch_size = hparams["batch_size"]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # initialize and load model
        self.model = load_model(hparams).to(self.device)
        self.model.load_from_checkpoint(checkpoint_path=self.path_to_model)
        # have to set to eval here because batch norm
        # has no meaning for one instance
        self.model.eval()

        self.dataset_cluster_ids = None
        self.clustering = get_clustering_alg(hparams)

        # dataset and transformations are the only
        # things we need from the DataModule
        self.image_dataloader = DataLoader(
            DataModule.dataset, batch_size=self.batch_size
        )
        self.transformation = DataModule.train_set_transformation

    def setup(self):
        """Setup the ImageSimilaritySearch class."""
        self.fit(self.image_dataloader)

        # cannot have the cluster ids loaded since the model cannot be used
        # if self.dataset_cluster_ids is None:
        self.dataset_cluster_ids = self.predict(
            dataloader=self.image_dataloader
        )
        self.save_cluster_ids_to_file(self.path_cluster_ids)
        # else:
        #     self.load_cluster_ids_from_file(self.path_cluster_ids)
        print(">>> setup completed successfully!")

    def save_cluster_ids_to_file(
        self, path: str = "./dataset_cluster_ids.csv"
    ):
        """Save the cluster ids to a file.

        Parameters
        ----------
        path : str, optional
            Path to the file. The default is "./dataset_cluster_ids.csv".
        """
        self.dataset = pd.DataFrame(
            self.image_dataloader.dataset.samples,
            columns=["image_path", "class_label"],
        )
        self.dataset["cluster_id"] = self.dataset_cluster_ids
        self.dataset.to_csv(path, index=False)
        print(">>> saved cluster ids to file")

    def load_cluster_ids_from_file(
        self, path: str = "./dataset_cluster_ids.csv"
    ):
        """Load the cluster ids from a file.

        Parameters
        ----------
        path : str, optional
            Path to the file. The default is "./dataset_cluster_ids.csv".
        """
        self.dataset = pd.read_csv(path)
        self.dataset_cluster_ids = self.dataset["cluster_id"].values
        print(">>> loaded cluster ids from file")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess the image.

        Parameters
        ----------
        image_path : str
            path to the image

        Returns
        -------
        torch.Tensor
            preprocessed image
        """
        return self.transformation(Image.open(image_path))

    def extract_features(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Extract features from the image.

        Parameters
        ----------
        image : torch.Tensor
            image to extract features

        Returns
        -------
        torch.Tensor
            features extracted from the image
        """
        # in case of no batch dimension is available
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)  # to cuda if available
        features = self.model.extract_features(image)
        features = features.detach()
        return features

    def fit(self, dataloader: DataLoader):
        """Fit the clustering algorithm.

        Parameters
        ----------
        dataloader : DataLoader
            dataloader can be train, validation or test dataloader
        """
        for batch_images, batch_labels in dataloader:
            features = self.extract_features(batch_images)
            features = features.cpu().numpy()
            self.clustering.partial_fit(features)

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """Predict the clusters for batch of images in the dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            dataloader can be train, validation or test dataloader

        Returns
        -------
        np.ndarray
            predicted clusters ids
        """
        predicted_clusters = []
        for batch_images, batch_labels in dataloader:
            features = self.extract_features(batch_images)
            features = features.cpu().numpy()
            _ = self.clustering.predict(features)
            predicted_clusters.extend(_)
        return np.array(predicted_clusters)

    def predict_one_image(self, image: torch.Tensor) -> int:
        """Predict the clusters for one image.

        Parameters
        ----------
        image : torch.Tensor
            image to predict the clusters

        Returns
        -------
        int
            predicted cluster id
        """
        # model needs the image to be a batch of size 1
        features = self.extract_features(image)
        features = features.cpu().numpy()
        cluster_id = self.clustering.predict(features)
        return cluster_id.item()

    def find_similar_images(
        self, path_to_image: str, topk: int = 5
    ) -> pd.DataFrame:
        """Find similar images to a given image.

        Parameters
        ----------
        path_to_image : str
            Path to the image.
        topk : int, optional
            Number of similar images to return. The default is 5.

        Returns
        -------
        pd.DataFrame
            A DataFrame object containing similar images.
        """
        image = self.preprocess_image(path_to_image)
        target_cluster_id = self.predict_one_image(image)
        query_indices = self.dataset["cluster_id"] == target_cluster_id
        query_image_paths = self.dataset[query_indices]["image_path"].values

        if topk:
            query_indices = self.sort_query_result(
                image, query_image_paths, topk
            )
            return self.dataset.iloc[query_indices]
        else:
            return self.dataset[query_indices]

    def sort_query_result(
        self,
        image: torch.Tensor,
        query_image_paths: list,
        topk: int = 5,
    ) -> list:
        """Sort the query result.

        Parameters
        ----------
        image : torch.Tensor
            image to predict the clusters
        query_image_paths : list
            list of image paths
        topk : int, optional
            Number of images to return. The default is 5.

        Returns
        -------
        list
            list of indices of the images in the dataset.
        """
        dist = torch.Tensor([]).to(self.device)
        image = self.extract_features(image)
        for batch_images in batchify(query_image_paths, self.batch_size):
            # batch_images is a list of image paths process them first
            processed_targets = [
                self.preprocess_image(path) for path in batch_images
            ]
            # convert to tensor
            processed_targets = torch.stack(processed_targets)
            z_targets = self.extract_features(processed_targets)
            # compute distance
            d = torch.cdist(image, z_targets).squeeze()
            # in case d is a scalar, add a dimension
            # it can cause an error with torch.cat if d is a scalar
            if len(query_image_paths) == 1:
                d = d.unsqueeze(0)
            dist = torch.cat((dist, d))
        # select top k
        if topk and len(query_image_paths) > topk:
            values, indices = torch.topk(-dist, k=topk)
        else:
            values, indices = torch.topk(-dist, k=len(query_image_paths))
        return indices.cpu().numpy().tolist()
