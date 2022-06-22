import typing
from torchvision import transforms
from torchvision.datasets import Omniglot
from image_classification_simulation.data.data_loader import MyDataModule
from torch.utils.data import DataLoader


# NB: background=True selects the train set,
# background=False selects the test set
# It's the nomenclature from the original paper,
# we just have to deal with it


class OmniglotLoader(MyDataModule):  # pragma: no cover
    """Data module class that prepares dataset\
        parsers and instantiates data loaders.

    Parameters
    ----------
    MyDataModule : DataModule class
        Template DataModule class
    """

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
        self.num_unique_labels = 964
        hyper_params["num_classes"] = self.num_unique_labels
        if "num_workers" in hyper_params:
            self.num_workers = hyper_params["num_workers"]
        else:
            self.num_workers = 1
            print("Number of workers set to:", self.num_workers)

        if "image_size" in hyper_params:
            self.image_size = hyper_params["image_size"]
        else:
            self.image_size = 28
            print("image size set to:", self.image_size)

        self.train_set_transformation = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        self.test_set_transformation = transforms.Compose(
            [
                # Omniglot images have 1 channel,
                #  but our model will expect 3-channel images
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(
                    [int(self.image_size * 1.15), int(self.image_size * 1.15)]
                ),
                transforms.CenterCrop(self.image_size),
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
        # here, we will actually assign train/val
        #  datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = Omniglot(
                root=self.data_dir,
                background=True,
                transform=self.train_set_transformation,
                download=True,
            )
            self.val_set = Omniglot(
                root=self.data_dir,
                background=True,
                transform=self.train_set_transformation,
                download=True,
            )
        if stage == "test" or stage is None:
            self.test_set = Omniglot(
                root=self.data_dir,
                background=False,
                transform=self.test_set_transformation,
                download=True,
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


if __name__ == "__main__":
    # tests the dataloader module
    args = {"batch_size": 32, "image_size": 28}
    omniglot_loader = OmniglotLoader("./examples/data/", args)
    omniglot_loader.setup(stage="fit")
