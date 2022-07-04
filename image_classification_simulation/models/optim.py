import logging
import torch
from torch import optim


logger = logging.getLogger(__name__)


def load_optimizer(hyper_params: dict, model: object):  # pragma: no cover
    """Instantiate the optimizer.

    Parameters
    ----------
    hyper_params : dict
        hyper parameters from the config file
    model : object
        the model to optimize

    Returns
    -------
    optimizer : object
        the optimizer to use
    """
    optimizer_name = hyper_params["optimizer"]
    if "lr" in hyper_params:
        lr = hyper_params["lr"]
    else:
        lr = 0.001
    # __TODO__ fix optimizer list
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr)
    elif optimizer_name == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr)
    else:
        raise ValueError("optimizer {} not supported".format(optimizer_name))
    return optimizer


def load_loss(hyper_params: dict):  # pragma: no cover
    r"""Instantiate the loss.

    You can add some math directly in your docstrings,
    however don't forget the `r`
    to indicate it is being treated as restructured text.
    For example, an L1-loss can be defined as:

    .. math::
        \text{loss}(x, y) = \frac{1}{n} \sum_{i} z_{i}

    Parameters
    ----------
    hyper_params : dict
        hyper parameters from the config file

    Returns
    -------
    loss : obj
        The loss for the given model
    """
    loss_name = hyper_params["loss"]
    if loss_name == "L1":
        loss = torch.nn.L1Loss(reduction="sum")
    elif loss_name == "CrossEntropyLoss":
        loss = torch.nn.CrossEntropyLoss(reduction="sum")
    elif loss_name == "MSELoss":
        loss = torch.nn.MSELoss()
    else:
        raise ValueError("loss {} not supported".format(loss_name))
    return loss
