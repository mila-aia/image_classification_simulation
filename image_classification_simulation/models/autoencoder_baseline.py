import torch
import typing
from torch import nn
from image_classification_simulation.models.optim import load_loss
from image_classification_simulation.models.my_model import BaseModel


class ConvAutoEncoder(BaseModel):
    """Holds a simple conv. autoencoder model."""

    def __init__(self, hyper_params: typing.Dict[typing.AnyStr, typing.Any]):
        """Calls the parent class and sets up the necessary\
            artifacts and hyperparameters.

        Parameters
        ----------
        hyper_params : typing.Dict[typing.AnyStr, typing.Any]
            A dictionary of hyperparameters
        """
        super(ConvAutoEncoder, self).__init__()

        self.save_hyperparameters(
            hyper_params
        )  # they will become available via model.hparams

        self.loss_fn = load_loss(hyper_params)

        if "num_filters" in hyper_params:
            self.num_filters = hyper_params["num_filters"]
        else:
            self.num_filters = 32

        # defining network layers

        self.pooling = nn.MaxPool2d(2, 2)

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
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.deconv2 = nn.ConvTranspose2d(
            self.num_filters // 2,
            self.num_filters,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.deconv3 = nn.ConvTranspose2d(
            self.num_filters,
            hyper_params["num_channels"],
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pooling,
            self.conv2,
            nn.ReLU(),
            self.pooling,
            self.conv3,
            nn.ReLU(),
            self.pooling,
        )

        self.decoder = nn.Sequential(
            self.deconv1,
            nn.ReLU(),
            self.deconv2,
            nn.ReLU(),
            self.deconv3,
            nn.Sigmoid(),
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
        input_data, _ = batch  # we don't need the targets
        reconstructed_input = self(
            input_data
        )  # calls the forward pass of the model
        loss = self.loss_fn(reconstructed_input, input_data)
        return loss, reconstructed_input

    def compute_reconstruction_similarity(
        self, input: torch.tensor, reconstructed_input: torch.tensor
    ) -> int:
        """Computes similarity between input img and recons. output.

        Parameters
        ----------
        input : torch.tensor
            input batch image.
        reconstructed_input : torch.tensor
            output batch image reconstucted by the autoencoder.

        Returns
        -------
        int
            The similarity score.
        """
        similarity = torch.abs(input - reconstructed_input)
        similarity = similarity.sum()
        return similarity

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
        loss, reconstructed_input = self._generic_step(batch, batch_idx)
        input_data, _ = batch
        train_metric = self.compute_reconstruction_similarity(input_data, reconstructed_input)
        self.log("train_loss", loss)
        self.log("train_similarity", train_metric)
        self.log("epoch", self.current_epoch)
        self.log("step", self.global_step)
        return {"loss": loss, "acc": train_metric}

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
        loss, reconstructed_input = self._generic_step(batch, batch_idx)
        input_data, _ = batch
        val_metric = self.compute_reconstruction_similarity(
            reconstructed_input, input_data
        )
        self.log("val_loss", loss)
        self.log("val_similarity", val_metric)
        return val_metric

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
        loss, reconstructed_input = self._generic_step(batch, batch_idx)
        input_data, _ = batch
        test_metric = self.compute_reconstruction_similarity(
            reconstructed_input, input_data
        )
        self.log("test_loss", loss)
        self.log("test_acc", test_metric)
        return test_metric

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

        bottleneck = self.encoder(batch_images)
        reconstructed_input = self.decoder(bottleneck)

        return reconstructed_input


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams = {
        "loss": "MSELoss",
        "optimizer": "adam",
        "num_channels": 3,
    }
    model = ConvAutoEncoder(hparams).to(device)
    print(model)
    # generate a random image to test the module
    img = torch.rand((16, 3, 224, 224)).to(device)
    labels = torch.randint(0, 31, (16,)).to(device)
    output = torch.rand((16, 3, 224, 224)).to(device)

    loss = model.training_step((img, labels), None)
    print(loss)

    similarity = model.compute_reconstruction_similarity(img, output)
    print(similarity)
