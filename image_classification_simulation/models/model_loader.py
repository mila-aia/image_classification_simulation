import logging

from image_classification_simulation.models.my_model import MyModel
from image_classification_simulation.models.resnet_baseline import Resnet


logger = logging.getLogger(__name__)


def load_model(hyper_params):  # pragma: no cover
    """Instantiate a model.

    Args:
        hyper_params (dict): hyper parameters from the config file

    Returns:
        model (obj): A neural network model object.
    """
    architecture = hyper_params["architecture"]
    # __TODO__ fix architecture list
    if architecture == "my_model":
        model_class = MyModel
    elif architecture == "resnet":
        model_class = Resnet
    else:
        raise ValueError("architecture {} not supported".format(architecture))
    logger.info("selected architecture: {}".format(architecture))

    model = model_class(hyper_params)
    logger.info("model info:\n" + str(model) + "\n")

    return model
