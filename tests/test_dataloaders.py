import pytest
import torch
from image_classification_simulation.data.mnist_loader import MNISTLoader


def generate_mnist_dataloader():
    args = {"batch_size": 16, "image_size": 28}
    mnist_loader = MNISTLoader("./examples/data/", args)
    mnist_loader.setup(stage="fit")
    return mnist_loader


def test_mnist_dataloader_features_shape():
    mnist_loader = generate_mnist_dataloader()
    train_features, _ = next(iter(mnist_loader.train_dataloader()))
    with pytest.raises(ValueError):
        assert train_features.size() == torch.Size([16, 1, 28, 28])


def test_mnist_dataloader_labels_shape():
    mnist_loader = generate_mnist_dataloader()
    _, train_labels = next(iter(mnist_loader.train_dataloader()))
    with pytest.raises(ValueError):
        assert train_labels.size() == torch.Size([16])
