import re
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors

class Clustering():
    def __init__(self, feature_extractor,hparams:dict):
        """Initialize the class.

        Parameters
        ----------
        feature_extractor : function
            Function that extracts features from an image.
        hparams : dict
            Hyper-parameters for the class.
        """
        self.model = self.get_clustering_alg(hparams)
        self.batch_size = hparams["batch_size"]
        self.extract_features = feature_extractor

    def get_clustering_alg(self, hparams: dict):
        """Initialize the class.

        Parameters
        ----------
        hparams : dict
            hyperparameters
        """
        self.alg_type = hparams["clustering_alg"]
        if self.alg_type == "MiniBatchKMeans":
            model = MiniBatchKMeans(
                n_clusters=hparams["num_clusters"],
                random_state=hparams["random_state"],
                batch_size=hparams["clustering_batch_size"],
                reassignment_ratio=hparams["reassignment_ratio"],
                verbose=1,
            )
        if self.alg_type == "BIRCH":
            model = Birch(
                # threshold=12.5,
                n_clusters=hparams["num_clusters"]
            )
        if self.alg_type == "knn":
            model = KNeighborsClassifier(
                n_neighbors=hparams["num_neighbors"],
                weights="distance",
                metric="precomputed",
            )
        if self.alg_type == "nn":
            model = NearestNeighbors(
                n_neighbors=hparams["num_neighbors"],
                radius=hparams["radius"],
                n_jobs=hparams["n_jobs"],
                metric="precomputed",
            )
        return model

    def get_type(self):
        """Get the type of the clustering algorithm."""
        return self.alg_type
    
    def fit(self,X : np.array,y : np.array =None):
        """Fit the clustering algorithm.

        Parameters
        ----------
        X : np.array
            Array of features.
        y : np.array
            Array of labels.
        """
        if self.dist_matrix is not None:
            self.model.fit(self.dist_matrix,y)
        else:
            self.model.fit(X)
        pass

    def build_dist_matrix(
            self,
            features: torch.Tensor,
            ) -> torch.Tensor: 
        """Build the distance matrix.

        Parameters
        ----------
        features : torch.Tensor
            Array of features.

        Returns
        -------
        torch.Tensor
            Distance matrix.
        """

        self.features = features
        self.dist_matrix = torch.cdist(features, features)
        return self.dist_matrix

    def fit_loader(self, dataloader: DataLoader):
        """Fit the clustering algorithm.

        Parameters
        ----------
        dataloader : DataLoader
            dataloader can be train, validation or test dataloader
        """
        for batch_images, batch_labels in dataloader:
            features = self.extract_features(batch_images)
            features = features.cpu().numpy()
            self.model.partial_fit(features)

        
    def predict(self, imgs: torch.Tensor)->int:  
        """Predict the clusters for images.

        Parameters
        ----------
        X : torch.Tensor
            Array of images.

        Returns
        -------
        int
            predicted cluster id
        """
        z_X = self.extract_features(imgs).cpu()
        if hasattr(self,'features'):
            dist = torch.cdist(z_X, self.features)
            predicted_label = self.model.predict(dist)
        else:
            predicted_label = self.model.predict(z_X.numpy())
        return predicted_label.item()

    def predict_loader(self, dataloader: DataLoader) -> np.ndarray:
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
        predicted_labels = []
        for batch_images, batch_labels in dataloader:
            features = self.extract_features(batch_images)
            features = features.cpu()  # .numpy()
            if hasattr(self,'features'):
                dist = torch.cdist(features, self.features)
                label = self.model.predict(dist)
            else:
                label = self.model.predict(features.numpy())
            predicted_labels.extend(label)
        return np.array(predicted_labels)

    def find_neighbors(
        self,
        imgs: torch.Tensor,
        k: int =5
        ):  
        """Find the neighbors for images.

        Parameters
        ----------
        imgs : torch.Tensor
            Array of images.
        k : int, optional
            number of neighbors, by default 5

        Returns
        -------
        np.ndarray
            neighbors ids
        """
        z_X = self.extract_features(imgs).cpu()
        dist = torch.cdist(z_X, self.features)
        label = self.model.kneighbors(
                    dist,
                    k,
                    return_distance=False
                    )
        return label[0]

    def get_inertia(self):
        """Get the inertia of the clustering algorithm."""
        return self.model.inertia_
