import typing
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from image_classification_simulation.data.data_loader import MyDataModule
from transformers import ViTFeatureExtractor


class Office31Loader(MyDataModule):  # pragma: no cover
    """Data module class.
    Prepares dataset parsers and instantiates data loaders.
    """

    # We are going to use the amazon data
    # (most similar to a catalog of online products)
    def __init__(
        self,
        data_dir: typing.AnyStr,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Validates the hyperparameter config dictionary and\
             sets up internal attributes.
        Parameters
        ----------
        data_dir : string
            Directory path that the data will be downloaded and stored
        hyper_params : dictionary
            Hyperparameters relevant to the dataloader module.
        """
        super().__init__(data_dir, hyper_params)
        self.num_unique_labels = 31
        hyper_params["num_classes"] = self.num_unique_labels

        if "num_workers" in hyper_params:
            self.num_workers = hyper_params["num_workers"]
        else:
            self.num_workers = 1
            print("Number of workers set to:", self.num_workers)

        if "image_size" in hyper_params:
            self.image_size = hyper_params["image_size"]
        else:
            self.image_size = 224
            print("image size set to:", self.image_size)

        self.train_set_transformation = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

        self.test_set_transformation = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage: str = None):
        """Parses and splits all samples across the train/valid/test parsers.
        Parameters
        ----------
        stage : string, optional
            Stage of training (training, validation, testing), by default None
        """
        # here, we will actually assign
        # train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            self.train_set = ImageFolder(
                root=self.data_dir, transform=self.train_set_transformation
            )

            n_val = int(np.floor(0.1 * len(self.train_set)))
            n_train = len(self.train_set) - n_val

            self.train_set, self.val_set = random_split(
                self.train_set, [n_train, n_val]
            )

        if stage == "test" or stage is None:

            self.test_set = ImageFolder(
                root=self.data_dir, transform=self.test_set_transformation
            )

    def train_dataloader(self) -> DataLoader:
        """Creates the training dataloader using the training data parser.
        Returns
        -------
        DataLoader
            returns a pytorch DataLoader class
        """
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            batch_sampler=None,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=None,
        )

    def val_dataloader(self) -> DataLoader:
        """Creates the validation dataloader using the validation data parser.
        Returns
        -------
        DataLoader
            returns a pytorch DataLoader class
        """
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            batch_sampler=None,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=None,
        )

    def test_dataloader(self) -> DataLoader:
        """Creates the testing dataloader using the testing data parser.
        Returns
        -------
        DataLoader
            returns a pytorch DataLoader class
        """
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            batch_sampler=None,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=None,
        )


class Office31LoaderViT(Office31Loader):  # pragma: no cover
    """Data module class.

    Prepares dataset parsers and instantiates data loaders.
    """

    # We are going to use the amazon data
    # (most similar to a catalog of online products)
    def __init__(
        self,
        data_dir: typing.AnyStr,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Validates the hyperparameter config dictionary and\
             sets up internal attributes.

        Parameters
        ----------
        data_dir : string
            Directory path that the data will be downloaded and stored
        hyper_params : dictionary
            Hyperparameters relevant to the dataloader module.
        """
        super().__init__(data_dir, hyper_params)
        self.num_unique_labels = 31
        hyper_params["num_classes"] = self.num_unique_labels

        if "num_workers" in hyper_params:
            self.num_workers = hyper_params["num_workers"]
        else:
            self.num_workers = 1

        if "train_test_split" in hyper_params:
            self.train_test_split = hyper_params["train_test_split"]
        else:
            self.train_test_split = 0.15

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

        self.train_set_transformation = transforms.Compose(
            [
                transforms.Resize(self.feature_extractor.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.feature_extractor.image_mean,
                    std=self.feature_extractor.image_std,
                ),
            ]
        )

        self.val_set_transformation = transforms.Compose(
            [
                transforms.Resize(self.feature_extractor.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.feature_extractor.image_mean,
                    std=self.feature_extractor.image_std,
                ),
            ]
        )


# import matplotlib as plt
if __name__ == "__main__":
    # tests the dataloader module
    args = {"batch_size": 8}
    office31_loader = Office31LoaderViT(
        "./examples/data/domain_adaptation_images/amazon/images", args
    )
    office31_loader.setup(stage="fit")
    i = iter(office31_loader.train_set.dataset)
    img, label = next(i)
    trans = transforms.ToPILImage()
    plt.imshow(trans(img))
    # plt.savefig('test.png')
    print(office31_loader)
