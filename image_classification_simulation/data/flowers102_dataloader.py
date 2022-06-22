import typing
from torchvision.datasets import Flowers102
from torchvision import transforms
from image_classification_simulation.data.data_loader import MyDataModule
from torch.utils.data import DataLoader


# NB: background=True selects the train set,
# background=False selects the test set
# It's the nomenclature from the original paper,
# we just have to deal with it


class Flowers102DataLoader(MyDataModule):  # pragma: no cover
    def __init__(
        self,
        data_dir: typing.AnyStr,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):

        super().__init__(data_dir, hyper_params)
        self.num_unique_labels = 102

        if "num_workers" in hyper_params:
            self.num_workers = hyper_params["num_workers"]
        else:
            self.num_workers = 1
            print("Number of workers set to:", self.num_workers)

        if "image_size" in hyper_params:
            self.image_size = hyper_params["image_size"]
        else:
            self.image_size = 128
            print("image size set to:", self.image_size)

        self.train_set_transformation = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.test_set_transformation = transforms.Compose(
            [
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
        # datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = Flowers102(
                root=self.data_dir,
                split="train",
                transform=None,
                target_transform=None,
                download=True,
            )
            self.val_set = Flowers102(
                root=self.data_dir,
                split="val",
                transform=None,
                target_transform=None,
                download=True,
            )
        if stage == "test" or stage is None:
            self.test_set = Flowers102(
                root=self.data_dir,
                split="test",
                transform=None,
                target_transform=None,
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
    flowers_loader = Flowers102DataLoader("./examples/data/", args)
    flowers_loader.setup(stage="fit")
