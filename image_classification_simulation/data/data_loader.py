import typing

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from image_classification_simulation.data.utils import get_data

# __TODO__ change the dataloader to suit your needs...


class MyDataset(Dataset):  # pragma: no cover
    """Dataset class for iterating over the data."""

    def __init__(
        self, input_data: np.ndarray, target_data: np.ndarray,
    ):
        """Initialize MyDataset.

        Args:
            input_data (np.array): Input data.
            target_data (np.array): Target data.
        """
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        """Return the number of data items in MyDataset."""
        return len(self.input_data)

    def __getitem__(
        self, index: int,
    ):
        """__getitem__.

        Args:
            index (int): Get index item from the dataset.
        """
        target_example = self.target_data[index]
        input_example = self.input_data[index]
        return input_example, target_example


class MyDataModule(pl.LightningDataModule):  # pragma: no cover
    """Data module class that prepares dataset
    parsers and instantiates data loaders."""

    def __init__(
        self,
        data_dir: typing.AnyStr,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Validates the hyperparameter config dictionary and
         sets up internal attributes."""
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = hyper_params["batch_size"]
        self.train_data_parser, self.dev_data_parser = None, None

    def prepare_data(self):
        """Downloads/extracts/unpacks the data if needed (we don't)."""
        pass

    def setup(self, stage=None):
        """Parses and splits all samples \
            across the train/valid/test parsers."""
        # here, we will actually assign
        # train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_input, train_target = get_data(self.data_dir, "train")
            self.train_data_parser = MyDataset(train_input, train_target)
            dev_input, dev_target = get_data(self.data_dir, "dev")
            self.dev_data_parser = MyDataset(dev_input, dev_target)
        if stage == "test" or stage is None:
            # __TODO__: add code to instantiate the test data parser here
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        """Creates the training dataloader using the training data parser."""
        return DataLoader(
            self.train_data_parser, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        """Creates the validation dataloader\
             using the validation data parser."""
        return DataLoader(
            self.dev_data_parser, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        """Creates the testing dataloader using the testing data parser."""
        # __TODO__: add code to instantiate the test data loader here
        raise NotImplementedError
