from image_classification_simulation.data.office31_loader import Office31Loader
from image_classification_simulation.data.omniglot_loader import OmniglotLoader
from image_classification_simulation.data.data_loader import MyDataModule


def load_data(data_dir: str, hyper_params: dict):  # pragma: no cover
    """Prepare the data into datasets.

    Parameters
    ----------
    data_dir : string
        path to the folder containing the data
    hyper_params : dictionary
        hyper parameters from the config file

    Returns
    -------
    DataModule
        Data module used to prepare/instantiate data loaders.
    """
    # __TODO__ if you have different data modules,
    #  add whatever code is needed to select them here
    if hyper_params["data"] == "Omniglot":
        return OmniglotLoader(data_dir, hyper_params)
    elif hyper_params["data"] == "Office31":
        return Office31Loader(data_dir, hyper_params)
    else:
        return MyDataModule(data_dir, hyper_params)


if __name__ == "__main__":
    data_dir = "./data"
    hyper_params = {
        "data": "Office31",
        "batch_size": 32,
        "epochs": 10,
        "seed": None,
        "lr": 0.001,
        "optimizer": "Adam",
        "patience": 3,
        "exp_name": "Office31",
    }
    load_data(data_dir, hyper_params)
