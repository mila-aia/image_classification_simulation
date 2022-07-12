import torch
import typing
import numpy as np
from torch import nn
from image_classification_simulation.utils.hp_utils import check_and_log_hp
from image_classification_simulation.models.optim import load_loss
from image_classification_simulation.models.my_model import BaseModel


class ClassicCNN(BaseModel):
    """Holds a simple standard CNN model."""

    def __init__(self, hyper_params: typing.Dict[typing.AnyStr, typing.Any]):
        """Calls the parent class and sets up the necessary\
            artifacts and hyperparameters.

        Parameters
        ----------
        hyper_params : typing.Dict[typing.AnyStr, typing.Any]
            A dictionary of hyperparameters
        """
        super(ClassicCNN, self).__init__()
        expected_hparams = {"num_classes", "img_size", "num_channels"}
        check_and_log_hp(expected_hparams, hyper_params)

        self.save_hyperparameters(
            hyper_params
        )  # they will become available via model.hparams

        self.loss_fn = load_loss(hyper_params)

        if "num_filters" in hyper_params:
            self.num_filters = hyper_params["num_filters"]
        else:
            self.num_filters = 8

        if "dropout_value" in hyper_params:
            self.dropout_value = hyper_params["dropout_value"]
        else:
            self.dropout_value = 0.1

        # defining network layers
        self.flatten = nn.Flatten()
        self.activation = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(self.dropout_value)

        self.conv1 = nn.Conv2d(
            hyper_params["num_channels"],
            self.num_filters,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            self.num_filters,
            self.num_filters * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            self.num_filters * 2,
            self.num_filters * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv4 = nn.Conv2d(
            self.num_filters * 4,
            self.num_filters * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv5 = nn.Conv2d(
            self.num_filters * 4,
            self.num_filters * 8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv6 = nn.Conv2d(
            self.num_filters * 8,
            self.num_filters * 8,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.bn1 = nn.BatchNorm2d(self.num_filters)
        self.bn2 = nn.BatchNorm2d(self.num_filters * 2)
        self.bn3 = nn.BatchNorm2d(self.num_filters * 4)
        self.bn4 = nn.BatchNorm2d(self.num_filters * 4)
        self.bn5 = nn.BatchNorm2d(self.num_filters * 8)
        self.bn6 = nn.BatchNorm2d(self.num_filters * 8)

        def get_output_shape(model, image_dim):
            return model(torch.rand(*(image_dim))).data.shape

        # Calculate the input size after the flatten layer
        self.expected_input_shape = (
            1,
            hyper_params["num_channels"],
            hyper_params["img_size"],
            hyper_params["img_size"],
        )

        conv1_out = get_output_shape(self.conv1, self.expected_input_shape)
        conv2_out = get_output_shape(
            self.maxpooling, get_output_shape(self.conv2, conv1_out)
        )
        conv3_out = get_output_shape(self.conv3, conv2_out)
        conv4_out = get_output_shape(
            self.maxpooling, get_output_shape(self.conv4, conv3_out)
        )
        conv5_out = get_output_shape(self.conv5, conv4_out)
        conv6_out = get_output_shape(
            self.maxpooling, get_output_shape(self.conv6, conv5_out)
        )

        fc_size = np.prod(list(conv6_out))

        self.linear1 = nn.Linear(fc_size, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, hyper_params["num_classes"])

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
        loss = self.loss_fn(logits, targets)
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

        # Block 1
        z_x = self.conv1(batch_images)
        z_x = self.bn1(z_x)
        z_x = self.activation(z_x)
        z_x = self.dropout(z_x)
        z_x = self.conv2(z_x)
        z_x = self.bn2(z_x)
        z_x = self.activation(z_x)
        z_x = self.dropout(z_x)
        z_x = self.maxpooling(z_x)

        # Block 2
        z_x = self.conv3(z_x)
        z_x = self.bn3(z_x)
        z_x = self.activation(z_x)
        z_x = self.dropout(z_x)
        z_x = self.conv4(z_x)
        z_x = self.bn4(z_x)
        z_x = self.activation(z_x)
        z_x = self.dropout(z_x)
        z_x = self.maxpooling(z_x)

        # Block 3
        z_x = self.conv5(z_x)
        z_x = self.bn5(z_x)
        z_x = self.activation(z_x)
        z_x = self.dropout(z_x)
        z_x = self.conv6(z_x)
        z_x = self.bn6(z_x)
        z_x = self.activation(z_x)
        z_x = self.dropout(z_x)
        z_x = self.maxpooling(z_x)

        # Fully connected block
        z_x = self.flatten(z_x)
        z_x = self.linear1(z_x)
        z_x = self.activation(z_x)
        z_x = self.linear2(z_x)
        z_x = self.activation(z_x)
        logits = self.linear3(z_x)

        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams = {
        "num_classes": 10,
        "img_size": 300,
        "num_channels": 3,
        "loss": "CrossEntropyLoss",
        "pretrained": True,
    }
    model = ClassicCNN(hparams).to(device)
    print(model)
    # generate a random image to test the module
    img = torch.rand((3, 3, 300, 300)).to(device)
    label = torch.randint(0, 10, (3,)).to(device)
    print(model(img).shape)

    loss = model.training_step((img, label), None)
    print(loss)
