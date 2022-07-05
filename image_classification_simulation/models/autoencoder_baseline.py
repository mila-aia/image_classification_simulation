import torch
import typing
from torch import nn
from image_classification_simulation.utils.hp_utils import check_and_log_hp
from image_classification_simulation.models.optim import load_loss
from image_classification_simulation.models.my_model import BaseModel


class ConvAutoEncoder(BaseModel):
    """Holds a simple standard CNN model."""

    def __init__(self, hyper_params: typing.Dict[typing.AnyStr, typing.Any]):
        """Calls the parent class and sets up the necessary\
            artifacts and hyperparameters.

        Parameters
        ----------
        hyper_params : typing.Dict[typing.AnyStr, typing.Any]
            A dictionary of hyperparameters
        """
        super(ConvAutoEncoder, self).__init__()
        check_and_log_hp(["num_classes"], hyper_params)

        self.save_hyperparameters(
            hyper_params
        )  # they will become available via model.hparams

        self.loss_fn = load_loss(hyper_params)

        if "num_filters" in hyper_params:
            self.num_filters = hyper_params["num_filters"]
        else:
            self.num_filters = 32

        # defining network layers

        self.conv1 = nn.Conv2d(
            hyper_params["num_channels"],
            self.num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            self.num_filters,
            self.num_filters // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv3 = nn.Conv2d(
            self.num_filters // 2,
            self.num_filters // 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.deconv1 = nn.ConvTranspose2d(
            self.num_filters // 4,
            self.num_filters // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.deconv2 = nn.ConvTranspose2d(
            self.num_filters // 2,
            self.num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.deconv3 = nn.ConvTranspose2d(
            self.num_filters,
            hyper_params["num_channels"],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
        )

        self.decoder = nn.Sequential(
            self.deconv1,
            nn.ReLU(),
            self.deconv2,
            nn.ReLU(),
            self.deconv3,
            nn.Tanh(),
        )

    def _generic_step(self, batch: typing.Any, batch_idx: int) -> typing.Any:
        """Runs the prediction + evaluation step for training/validation/testing.

        Parameters
        ----------
        batch : typing.Any
            input batch of data
        batch_idx : int
            index of the input batch

        Returns
        -------
        typing.Any
            returns loss and logit scores.
        """
        input_data, targets = batch
        logits = self(input_data)  # calls the forward pass of the model
        print("targets shape: ", targets.shape)
        print("logits shape: ", logits.shape)
        loss = self.loss_fn(logits, input_data)
        return loss, logits

    def compute_accuracy(
        self, logits: torch.tensor, targets: torch.tensor
    ) -> int:
        """Computes accuracy given the logits and target labels.

        Parameters
        ----------
        logits : torch.tensor
            Scores produced by passing the input data to the model.
        targets : torch.tensor
            True labels of the data points.

        Returns
        -------
        int
            The accuracy score.
        """
        probs = nn.functional.softmax(logits, 0)
        preds = torch.argmax(probs, 1)
        return (preds == targets).sum().item() / len(targets)

    def training_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """Runs a prediction step for training, returning the loss.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of data (images).
        batch_idx : torch.Tensor
            A batch of data points' labels.

        Returns
        -------
        torch.Tensor
            loss produced by the loss function.
        """
        loss, logits = self._generic_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("epoch", self.current_epoch)
        self.log("step", self.global_step)
        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """Runs a prediction step for validation, logging the loss.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of data (images).
        batch_idx : torch.Tensor
            A batch of data points' labels.

        Returns
        -------
        torch.Tensor
            loss produced by the loss function.
        """
        loss, logits = self._generic_step(batch, batch_idx)
        input_data, targets = batch
        val_acc = self.compute_accuracy(logits, targets)
        self.log("val_loss", loss)
        self.log("val_acc", val_acc)

    def test_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """Runs a prediction step for testing, logging the loss.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of data (images).
        batch_idx : torch.Tensor
            A batch of data points' labels.

        Returns
        -------
        torch.Tensor
            loss produced by the loss function.
        """
        loss, logits = self._generic_step(batch, batch_idx)
        input_data, targets = batch
        test_acc = self.compute_accuracy(logits, targets)
        self.log("test_loss", loss)
        self.log("test_acc", test_acc)

    def forward(self, batch_images: torch.Tensor) -> torch.Tensor:
        """Passes a batch of data to the model.

        Parameters
        ----------
        batch_images : torch.Tensor
            A batch of input images data.

        Returns
        -------
        torch.Tensor
            Logit scores
        """
        # print(batch_images.shape)

        z_x = self.encoder(batch_images)
        print("shape of code: ", z_x.shape)
        logits = self.decoder(z_x)

        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams = {
        "num_classes": 31,
        "loss": "MSELoss",
        "optimizer": "adam",
        "num_channels": 3,
    }
    model = ConvAutoEncoder(hparams).to(device)
    print(model)
    # generate a random image to test the module
    img = torch.rand((1, 3, 100, 100))
    label = torch.randint(0, 31, (3,))
    print(model(img).shape)

    loss = model.training_step((img, label), None)
    print(loss)
