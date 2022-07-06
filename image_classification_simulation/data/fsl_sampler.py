import random
from typing import List, Tuple, Iterator

import torch
from torch import Tensor
from torch.utils.data import Sampler
from torch.utils.data import DataLoader

# from easyfsl.datasets import FewShotDataset


class TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks.
    At each iteration, it will sample n_way classes, and
    then sample support and query images from these classes.
    """

    def __init__(
        self,
        dataset: DataLoader,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
    ):
        """Initializes the sampler.

        Parameters
        ----------
        dataset : DataLoader
            The dataset to sample from.
        n_way : int
            The number of classes per episode.
        n_shot : int
            The number of support images per class.
        n_query : int
            The number of query images per class.
        n_tasks : int
            The number of episodes to sample.
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks

        self.items_per_label = {}
        for item, label in enumerate(dataset.get_labels()):
            if label in self.items_per_label.keys():
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_tasks):
            s = []
            sampled_labels = random.sample(
                self.items_per_label.keys(), self.n_way
            )
            for label in sampled_labels:
                # samples = random.sample(self.items_per_label[label], self.n_shot + self.n_query)
                samples = random.choices(
                    population=self.items_per_label[label],
                    k=self.n_shot + self.n_query,
                )
                samples = torch.tensor(samples)
                s.append(samples)
            yield torch.cat(s).tolist()

    def episodic_collate_fn(
        self, input_data: List[Tuple[Tensor, int]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.

        Parameters
        ----------
        input_data : List[Tuple[Tensor, int]]
            A list of tuples containing the data and the label.

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]
            A tuple containing the support images, the support labels, the query images, the query labels,
            and the list of labels.
        """
        true_class_ids = list({x[1] for x in input_data})

        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        # pylint: disable=not-callable
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        # pylint: enable=not-callable

        support_images = all_images[:, : self.n_shot].reshape(
            (-1, *all_images.shape[2:])
        )
        query_images = all_images[:, self.n_shot :].reshape(
            (-1, *all_images.shape[2:])
        )
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )
