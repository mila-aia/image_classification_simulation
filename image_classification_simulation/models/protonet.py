import torch
import typing
from torch import nn
from torchvision.models import resnet18
from image_classification_simulation.models.optim import load_loss
from image_classification_simulation.models.my_model import BaseModel


class PrototypicalNetworks(BaseModel):
    """Prototypical Networks used for few-shot learning."""

    def __init__(self, hyper_params: typing.Dict[typing.AnyStr, typing.Any]):
        super(PrototypicalNetworks, self).__init__()
        self.save_hyperparameters(
            hyper_params
        )  # they will become available via model.hparams

        self.loss_fn = load_loss(hyper_params)

        # temporarily hard coded resnet18
        # it should be specified in the hyper_params
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Flatten()

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
        (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
        ) = batch
        scores = self(
            support_images, support_labels, query_images
        )  # calls the forward pass of the model
        loss = self.loss_fn(scores, query_labels)
        return loss, scores

    def compute_accuracy(self, scores, query_labels) -> int:
        """Returns the number of correct predictions of query labels,\
            and the total number of predictions."""
        preds = torch.max(
            scores.detach().data,
            1,
        )[1]
        return (preds == query_labels).sum().item(), len(query_labels)

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
        query_labels = batch[3]
        correct, total = self.compute_accuracy(logits, query_labels)
        self.log("train_loss", loss)
        self.log("train_acc", correct / total)
        self.log("epoch", self.current_epoch)
        self.log("step", self.global_step)
        return {"loss": loss, "correct": correct, "total": total}

    def training_epoch_end(
        self, training_step_outputs: typing.List[float]
    ) -> None:
        """Is called at the end of each epoch.

        Parameters
        ----------
        training_step_outputs : typing.List[float]
            A list of training accuracy scores\
                    produced by the training step.
        """
        total_correct = sum([i["correct"] for i in training_step_outputs])
        total = sum([i["total"] for i in training_step_outputs])
        overal_train_loss = torch.stack(
            [i["loss"] for i in training_step_outputs]
        ).mean()
        self.log("overall_train_loss", overal_train_loss)
        self.log("overall_train_acc", total_correct / total)

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
        loss, scores = self._generic_step(batch, batch_idx)
        query_labels = batch[3]
        correct, total = self.compute_accuracy(scores, query_labels)
        self.log("val_loss", loss)
        self.log("val_acc", correct / total)
        return {"loss": loss, "correct": correct, "total": total}

    def validation_epoch_end(
        self, validation_step_outputs: typing.List[float]
    ) -> float:
        """Is called at the end of each epoch.

        Parameters
        ----------
        validation_step_outputs : typing.List[float]
            A list of validation accuracy scores\
                 produced by the validation step.

        Returns
        -------
        float
            The average validation accuracy over all batches.
        """
        total_correct = sum([i["correct"] for i in validation_step_outputs])
        total = sum([i["total"] for i in validation_step_outputs])
        overal_val_acc = total_correct / total
        overal_val_loss = torch.stack(
            [i["loss"] for i in validation_step_outputs]
        ).mean()
        self.log("overall_valid_loss", overal_val_loss)
        self.log("overall_valid_acc", overal_val_acc)
        return overal_val_acc

    def test_step(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> dict:
        """Runs a prediction step for testing, logging the loss.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of data (images).
        batch_idx : torch.Tensor
            A batch of data points' labels.

        Returns
        -------
        dict
            loss produced by the loss function.
            correct and total are the number of correct predictions
            and the total number of predictions.
        """
        loss, scores = self._generic_step(batch, batch_idx)
        query_labels = batch[3]
        correct, total = self.compute_accuracy(scores, query_labels)
        self.log("test_loss", loss)
        self.log("test_acc", correct / total)
        return {"loss": loss, "correct": correct, "total": total}

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """Predict query labels using labeled support images."""
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes
        # from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of
        # features corresponding to labels == i
        self.z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, self.z_proto)

        scores = -dists
        return scores


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams = {
        "num_workers": 2,
        "batch_size": 32,
        "n_way": 31,
        # with high number of classes we can't sample enough samples
        "n_shot": 10,
        # use lower number of samples for now
        # until a smarter data spliting is devised
        "n_query": 10,
        "num_training_episodes": 400,
        "num_eval_tasks": 50,
    }
    convolutional_network = resnet18(pretrained=True)
    convolutional_network.fc = nn.Flatten()
    model = PrototypicalNetworks(hparams).to(device)
