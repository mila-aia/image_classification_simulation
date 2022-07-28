import typing
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn import metrics


def get_clustering_metrics(
    labels_true: np.array, labels_predicted: np.array
) -> dict:
    """
    Computes the clustering metrics.

    Parameters
    ----------
    labels_true : np.array of shape (n_samples,)
        The true labels.
    labels_predicted : np.array, shape (n_samples)
        The predicted labels.
    """
    # https://scikit-learn.org/stable/modules/clustering.html#rand-score
    m = {
        "rand_score": metrics.adjusted_rand_score(
            labels_true, labels_predicted
        ),
        "adjusted_rand_score": metrics.adjusted_rand_score(
            labels_true, labels_predicted
        ),
        "mutual_info_score": metrics.mutual_info_score(
            labels_true, labels_predicted
        ),
    }
    return m


def show_grid_images(
    images: typing.List[torch.Tensor],
    label: list,
    height: int = 8,
    width: int = 8,
    save_path: str = None,
):
    """Show a grid of images.

    Parameters
    ----------
    images : _type_
        list of images
    label : _type_
        _description_
    height : int, optional
        image height, by default 8
    width : int, optional
        image width, by default 8
    save_path : str, optional
        path to save image, by default None
    """
    grid = make_grid(images)
    plt.figure(figsize=(height, width))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("cluster {}".format(label))
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)


def dataloader_to_images(
    dataloader: DataLoader,
) -> typing.Union[typing.List[torch.Tensor], list]:
    """Convert a dataloader to a list of images.

    Parameters
    ----------
    dataloader : DataLoader
        dataloader to convert to a list of images

    Returns
    -------
    typing.Union[typing.List[torch.Tensor], list]
        list of images and labels
    """
    images, labels = [], []
    for batch_images, batch_labels in dataloader:
        images.extend(batch_images.detach().cpu())
        labels.extend(batch_labels.detach().cpu().tolist())
    return images, labels


def show_images_in_clusters(
    images: typing.List[torch.Tensor], predicted_clusters: np.ndarray
):
    """Show images in clusters.

    Parameters
    ----------
    images : typing.List[torch.Tensor]
        list of images
    predicted_clusters : np.ndarray
        predicted clusters indices
    """
    num_clusters = np.unique(predicted_clusters)
    for num in num_clusters:
        print(
            "cluster {} has {} images".format(
                num, np.sum(predicted_clusters == num)
            )
        )
        images_in_cluster = np.array(images)[predicted_clusters == num]
        show_grid_images(images_in_cluster.tolist(), num)


def get_inertia(clustering_alg):
    """Get the inertia of the clustering algorithm."""
    return clustering_alg.inertia_


def visualize(dataloader: DataLoader, predicted_clusters):
    """Visualize the clusters.

    Parameters
    ----------
    dataloader : DataLoader
        dataloader can be train, validation or test dataloader
    """
    images, labels = dataloader_to_images(dataloader)
    # predicted_clusters = self.predict(dataloader)
    show_images_in_clusters(images, predicted_clusters)
    return predicted_clusters


def evaluate_performance(
    class_ids: np.array,
    true_labels: list,
    find_topk: typing.Callable,
    dataloader: typing.Iterable,
    topk: int,
) -> float:
    """Evaluate the performance of the model.

    Parameters
    ----------
    class_ids : np.array
        list of class ids from the whole dataset
    true_labels : list
        list of true class ids from the evaluation dataset
    find_topk : typing.Callable
        function to find top k similar images
    dataloader : typing.Iterable
        dataloader used to evaluate the performance
    topk : int
        number of top k to evaluate

    Returns
    -------
    float
        accuracy of the model
    """
    topk_sim_imgs = [
        find_topk(b_imgs, topk) for b_imgs, b_labels in dataloader
    ]
    # topk_sim_imgs: are ids of topk similar images
    topk_sim_imgs = np.concatenate(topk_sim_imgs)
    class_ids = np.array(class_ids)
    # get class ids of the topk similar images from their data ids
    pred_class_ids = [
        class_ids[topk_sim_img] for topk_sim_img in topk_sim_imgs
    ]
    pred_class_ids = np.stack(pred_class_ids)
    total_num_match = 0
    for pred_class_id, true_label in zip(pred_class_ids, true_labels):
        total_num_match += (true_label == pred_class_id).sum()
    return total_num_match / (len(true_labels) * topk)
