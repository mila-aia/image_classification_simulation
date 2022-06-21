import os
import typing
import numpy as np
from image_classification_simulation.data.office31_loader import Office31Loader
from image_classification_simulation.data.omniglot_loader import OmniglotLoader
from image_classification_simulation.data.data_loader import MyDataModule


def load_data(data_dir, hyper_params):  # pragma: no cover
    """Prepare the data into datasets.

    Args:
        data_dir (str): path to the folder containing the data
        hyper_params (dict): hyper parameters from the config file

    Returns:
        datamodule (obj):
        the data module used to prepare/instantiate data loaders.
    """
    # __TODO__ if you have different data modules,
    #  add whatever code is needed to select them here
    if hyper_params['data'] == 'Omniglot':
        return OmniglotLoader(data_dir, hyper_params)
    if hyper_params['data'] == 'Office31':
        return Office31Loader(data_dir, hyper_params)
    else:
        return MyDataModule(data_dir, hyper_params)