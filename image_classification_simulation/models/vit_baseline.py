import torch
import typing
from transformers import ViTForImageClassification
from image_classification_simulation.utils.hp_utils import check_and_log_hp
from image_classification_simulation.models.optim import load_loss
from image_classification_simulation.models.my_model import BaseModel


class ViT(BaseModel):
    """Loads a pretrained ViT baseline (from HuggingFace) and finetunes it."""

    def __init__(self, hyper_params: typing.Dict[typing.AnyStr, typing.Any]):
        """Calls the parent class and sets up the necessary\
            artifacts and hyperparameters.

        Parameters
        ----------
        hyper_params : typing.Dict[typing.AnyStr, typing.Any]
            A dictionary of hyperparameters
        """
        super(ViT, self).__init__()
        check_and_log_hp(["num_classes"], hyper_params)

        self.save_hyperparameters(
            hyper_params
        )  # they will become available via model.hparams

        self.loss_fn = load_loss(hyper_params)

        self.num_classes = hyper_params["num_classes"]

        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=self.num_classes
        )

        layers = list(self.vit.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*layers)
        num_target_classes = hyper_params["num_classes"]
        dim_features = 768  # 3 channels * (16*16) patches in the ViT model
        self.linear = torch.nn.Linear(dim_features, num_target_classes)

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
        probs = torch.nn.functional.softmax(logits, 1)
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
        input_data, targets = batch
        train_acc = self.compute_accuracy(logits, targets)
        self.log("train_loss", loss)
        self.log("train_acc", train_acc)
        self.log("epoch", self.current_epoch)
        self.log("step", self.global_step)
        return {"loss": loss, "acc": train_acc}

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
        return val_acc

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
        return test_acc

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
        z_x = self.extract_features(batch_images)
        logits = self.linear(z_x)  # MLP head
        return logits

    def extract_features(self, batch_images: torch.Tensor):
        """Extracts features from the model.

        Returns
        -------
        torch.Tensor
            The extracted features.
        """
        z_x = self.feature_extractor(batch_images)
        z_x = z_x["last_hidden_state"]
        return z_x[:, 0, :]  # take cls token only


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams = {
        "loss": "CrossEntropyLoss",
        "pretrained": True,
        "num_classes": 31,
    }
    print(device)
    model = ViT(hparams).to(device)
    print(model)
    # generate a random image to test the module
    img = torch.rand((16, 3, 224, 224)).to(device)
    label = torch.randint(0, 31, (16,)).to(device)
    print(model(img).shape)

    loss = model.training_step((img, label), None)
    print(loss)
