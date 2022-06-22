import typing
from torchvision import transforms
import numpy as np
from torchvision.datasets import ImageFolder
from image_classification_simulation.data.data_loader import MyDataModule
from torch.utils.data import DataLoader, random_split


class Office31Loader(MyDataModule):  # pragma: no cover
    """Data module class that prepares dataset parsers and instantiates data loaders.

    Parameters
    ----------
    MyDataModule : DataModule class
        Template DataModule class
    """

    # We are going to use the amazon data
    # (most similar to a catalog of online products)

    def __init__(
        self,
        data_dir: typing.AnyStr,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Validates the hyperparameter config dictionary and sets up internal attributes.

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
            self.image_size = 200
            print("image size set to:", self.image_size)

        self.train_set_transformation = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((200, 200)),
                transforms.ToTensor(),
            ]
        )

        self.test_set_transformation = transforms.Compose(
            [transforms.Resize((200, 200)), transforms.ToTensor()]
        )

    def setup(self, stage: str = None):

        """Parses and splits all samples across the train/valid/test parsers.
        Parameters
        ----------
        stage : string, optional
            Stage of training (training, validation, testing), by default None
        """

        # here, we will actually assign train/val datasets for use in dataloaders
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


# import matplotlib as plt
if __name__ == "__main__":
    # tests the dataloader module
    args = {"batch_size": 8, "image_size": 200}
    office31_loader = Office31Loader("../../examples/data/amazon", args)
    office31_loader.setup(stage="fit")
    # i = iter(office31_loader.train_set.dataset)
    # img, label = next(i)
    # trans = transforms.ToPILImage()
    # plt.imshow(trans(img))
    # plt.savefig('test.png')
    # print(office31_loader)
