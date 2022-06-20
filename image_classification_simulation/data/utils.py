import os
import typing
import numpy as np
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
    else:
        return MyDataModule(data_dir, hyper_params)


def get_data(
    data_folder: typing.AnyStr,
    prefix: typing.AnyStr
) -> typing.Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    """Function to load data into memory.

    Args:
        data_folder (str): Path of the folder where the data lives.
        prefix (str): The data split to target, i.e. "train" or "dev.

    Returns:
        in_data (np.array): Input data.
        tar_data (np.array): Target data.
    """
    inputs = []
    with open(os.path.join(data_folder, '{}.input'.format(prefix))) as in_stream:
        for line in in_stream:
            inputs.append([float(x) for x in line.split()])
    in_data = np.array(inputs, dtype=np.float32)
    targets = []
    with open(os.path.join(data_folder, '{}.target'.format(prefix))) as in_stream:
        for line in in_stream:
            targets.append(float(line))
    tar_data = np.array(targets, dtype=np.float32)
    return in_data, tar_data
