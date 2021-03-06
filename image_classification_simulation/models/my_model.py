import logging
import typing
import torch
import pytorch_lightning as pl

from image_classification_simulation.models.optim import load_loss
from image_classification_simulation.models.optim import load_optimizer

from image_classification_simulation.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


class BaseModel(pl.LightningModule):
    """Base class for Pytorch Lightning model useful to reuse\
        the same *_step methods."""

    def configure_optimizers(self):
        """Returns the combination of optimizer(s)\
         and learning rate scheduler(s) to train with.

        Here, we read all the optimization-related
        hyperparameters from the config dictionary and
        create the required optimizer/scheduler combo.
        This function will be called automatically by
        the pytorch lightning trainer implementation.
        See
        https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html
        for more info on the expected returned elements.
        """
        # we use the generic loading function from the `model_loader` module,
        #  but it could be made a direct part of the model
        # (useful if we want layer-dynamic optimization)
        return load_optimizer(self.hparams, self)

    def _generic_step(
        self,
        batch: typing.Any,
        batch_idx: int,
    ) -> typing.Any:
        """Runs the prediction + evaluation\
        step for training/validation/testing."""
        input_data, targets = batch
        preds = self(input_data)  # calls the forward pass of the model
        loss = self.loss_fn(preds, targets)
        return loss

    def training_step(self, batch, batch_idx):
        """Runs a prediction step for training, returning the loss."""
        loss = self._generic_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("epoch", self.current_epoch)
        self.log("step", self.global_step)
        # this function is required,
        # as the loss returned here is used for backprop
        return loss

    def validation_step(self, batch, batch_idx):
        """Runs a prediction step for validation, logging the loss."""
        loss = self._generic_step(batch, batch_idx)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        """Runs a prediction step for testing, logging the loss."""
        loss = self._generic_step(batch, batch_idx)
        self.log("test_loss", loss)

    def extract_features(
        self, batch: torch.Tensor
    ) -> typing.Union[torch.Tensor, None]:
        """Extracts features from the model."""
        if hasattr(self, "feature_extractor"):
            z_x = self.feature_extractor.forward(batch)
            z_x = self.flatten(z_x)
            return z_x
        else:
            raise Exception("No feature extractor found.")


class MyModel(BaseModel):  # pragma: no cover
    """Simple Model Class.

    Inherits from the given framework's model class.
    This is a simple MLP model.
    """

    def __init__(self, hyper_params: typing.Dict[typing.AnyStr, typing.Any]):
        """Initialize the model.

        Parameters
        ----------
        hyper_params : typing.Dict[typing.AnyStr, typing.Any]
            A dictionary of hyperparameters
        """
        super(MyModel, self).__init__()

        check_and_log_hp(["size"], hyper_params)
        # they will become available via model.hparams
        self.save_hyperparameters(hyper_params)
        self.linear1 = torch.nn.Linear(5, hyper_params["size"])
        self.linear2 = torch.nn.Linear(hyper_params["size"], 1)
        # 'load_loss' could be part of the model itself...
        self.loss_fn = load_loss(hyper_params)

    def forward(self, data: torch.tensor) -> torch.tensor:
        """Forward method of the model.

        Parameters
        ----------
            data : torch.tensor
                The data to be passed to the model.

        Returns
        -------
        torch.tensor
            The output of the model.
        """
        hidden = torch.nn.functional.relu(self.linear1(data))
        result = self.linear2(hidden)
        return result.squeeze()
