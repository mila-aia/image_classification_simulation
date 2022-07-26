import logging
from image_classification_simulation.models.autoencoder_baseline import (
    ConvAutoEncoder,
)
from image_classification_simulation.models.my_model import MyModel
from image_classification_simulation.models.resnet_baseline import Resnet
from image_classification_simulation.models.vit_baseline import ViT
from image_classification_simulation.models.classic_cnn_baseline import (
    ClassicCNN,
)
from image_classification_simulation.models.protonet import (
    PrototypicalNetworks,
)


logger = logging.getLogger(__name__)


def load_model(hyper_params: dict):  # pragma: no cover
    """Instantiate a model.

    Parameters
    ----------
    hyper_params : dict
        hyper parameters from the config file

    Returns
    -------
    model : object
        the model to use
    """
    architecture = hyper_params["architecture"]
    # __TODO__ fix architecture list
    if architecture == "my_model":
        model_class = MyModel
    elif architecture == "resnet":
        model_class = Resnet
    elif architecture == "vit":
        model_class = ViT
    elif architecture == "classic-cnn":
        model_class = ClassicCNN
    elif architecture == "protonet":
        model_class = PrototypicalNetworks
    elif architecture == "conv_ae":
        model_class = ConvAutoEncoder
    else:
        raise ValueError("architecture {} not supported".format(architecture))
    logger.info("selected architecture: {}".format(architecture))

    model = model_class(hyper_params)
    logger.info("model info:\n" + str(model) + "\n")

    return model
