import os
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
from image_classification_simulation.models.clustering import Clustering

# import typing
# import numpy as np
# from joblib import dump, load


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
        batch_size = hparams["batch_size"]

        if os.path.exists(self.path_cluster_ids):
            self.load_cluster_ids_from_file(self.path_cluster_ids)
            print(">>> Found cluster ids from file")

        self.dataset_cluster_ids = None
        self.clustering = Clustering(hparams)

        # for now we only initialize the model
        # loading clustering alg with joblib is not working properly
        # methods are not loaded properly
        self.model_loaded = False
        # if os.path.exists(self.path_to_model):
        #     self.clustering.clustering_alg
        #  = self.clustering.load_model_from_file(
        #         self.path_to_model
        #     )
        #     self.model_loaded = True
        #     print(">>> Found and loaded model from file")
        # else:
        #     self.model_loaded = False
        #     print(">>> Initiated a new model!")

        # dataset and transformations are the only
        # things we need from the DataModule
        self.image_dataloader = DataLoader(
            DataModule.dataset, batch_size=batch_size
        )

        self.transformation = DataModule.train_set_transformation

    def setup(self):
        """Setup the ImageSimilaritySearch class."""
        if self.model_loaded is False:
            self.clustering.fit(self.image_dataloader)
            self.clustering.save_model_to_file(self.path_to_model)

        # cannot have the cluster ids loaded since the model cannot be used
        # if self.dataset_cluster_ids is None:
        self.dataset_cluster_ids = self.clustering.predict(
            dataloader=self.image_dataloader
        )
        self.save_cluster_ids_to_file(self.path_cluster_ids)
        # else:
        #     self.load_cluster_ids_from_file(self.path_cluster_ids)
        print(">>> setup completed successfully!")

    def save_cluster_ids_to_file(self, path="./dataset_cluster_ids.csv"):
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

    def load_cluster_ids_from_file(self, path="./dataset_cluster_ids.csv"):
        """Load the cluster ids from a file.

        Parameters
        ----------
        path : str, optional
            Path to the file. The default is "./dataset_cluster_ids.csv".
        """
        self.dataset = pd.read_csv(path)
        self.dataset_cluster_ids = self.dataset["cluster_id"].values
        print(">>> loaded cluster ids from file")

    def preprocess_image(self, image_path):
        """Preprocess the image."""
        return self.transformation(Image.open(image_path))

    def find_similar_images(self, path_to_image, topk=5) -> pd.DataFrame:
        """Find similar images to a given image.

        Parameters
        ----------
        path_to_image : str
            Path to the image.

        Returns
        -------
        pd.DataFrame
            A DataFrame object containing similar images.
        """
        image = self.preprocess_image(path_to_image)
        target_cluster_id = self.clustering.predict_one_image(image)
        query_indices = self.dataset["cluster_id"] == target_cluster_id

        query_image_paths = self.dataset[query_indices]["image_path"].values
        list_processed_imgs = [
            self.preprocess_image(path) for path in query_image_paths
        ]
        if topk:
            query_indices = self.clustering.compute_distances(
                image, list_processed_imgs, topk
            )
            return self.dataset.iloc[query_indices]
        else:
            return self.dataset[query_indices]
