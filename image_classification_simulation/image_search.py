import os
import typing
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from image_classification_simulation.models.clustering import Clustering

class Image_similarity_search():
    def __init__(self, hparams, DataModule) -> None:
        self.clustering = Clustering(hparams)

        batch_size = hparams["batch_size"]
        # dataset and trasnformations are the only 
        # things we need from the DataModule
        self.image_dataloader = DataLoader(
            DataModule.dataset,
            batch_size=batch_size
            )

        self.transformation = DataModule.train_set_transformation

        self.path_to_model = hparams["path_to_model"]
        self.path_cluster_ids = hparams["path_cluster_ids"]

        if os.path.exists( self.path_cluster_ids):
            self.load_cluster_ids_from_file(self.path_cluster_ids)

    def setup(self):
        if os.path.exists(self.path_to_model):
            self.clustering.load_model_from_file(self.path_to_model)
        else:
            self.clustering.fit(self.image_dataloader)
            if os.path.exists(self.path_cluster_ids):
                self.load_cluster_ids_from_file(self.path_cluster_ids)
            else:
                self.dataset_cluster_ids = self.clustering.predict(
                    dataloader=self.image_dataloader
                    )
                self.save_cluster_ids_to_file(self.path_cluster_ids)
        print('>>> setup completed successfully!')

    def save_cluster_ids_to_file(self, path='./cluster_ids.npy'):
        np.save(path,self.dataset_cluster_ids)
        print('>>> saved cluster ids to file')

    def load_cluster_ids_from_file(self, path='./cluster_ids.npy'):
        self.dataset_cluster_ids = np.load(path)
        print('>>> loaded cluster ids from file')

    def predict_image_cluster(self, path)->int:
        image = Image.open(path)
        image = self.transformation(image)
        return self.clustering.predict_one_image(image)

    def find_similar_images(
        self,
        path_to_image
        )->typing.Tuple[list,list]:
        target_cluster_id = self.predict_image_cluster(path_to_image)
        target_images = []
        target_cluster_ids = []
        for (image, class_label), cluster_id in zip(
            self.image_dataloader.dataset,
            self.dataset_cluster_ids
            ):
            if cluster_id == target_cluster_id:
                target_images.append(image)
                target_cluster_ids.append(cluster_id)
        return target_images, target_cluster_ids