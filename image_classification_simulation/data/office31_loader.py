import typing
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from image_classification_simulation.data.data_loader import MyDataModule
from easyfsl.samplers import TaskSampler


class Office31Loader(MyDataModule):  # pragma: no cover
    """Data module class.

    Prepares dataset parsers and instantiates data loaders.
    """

    # We are going to use the amazon data
    # (most similar to a catalog of online products)
    def __init__(
        self,
        data_dir: typing.AnyStr,
        hyper_params: typing.Dict[typing.AnyStr, typing.Any],
    ):
        """Validates the hyperparameter config dictionary and\
             sets up internal attributes.

        Parameters
        ----------
        data_dir : string
            Directory path that the data will be downloaded and stored
        hyper_params : dictionary
            Hyperparameters relevant to the dataloader module.
        """
        super().__init__(data_dir, hyper_params)
        self.num_unique_labels = 31
        hyper_params["num_classes"] = self.num_unique_labels

        if "num_workers" in hyper_params:
            self.num_workers = hyper_params["num_workers"]
        else:
            self.num_workers = 1

        if "image_size" in hyper_params:
            self.image_size = hyper_params["image_size"]
        else:
            self.image_size = 300

        self.train_set_transformation = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.ToTensor(),
            ]
        )

        self.test_set_transformation = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage: str = None, valid_size: float = 0.1):
        """Parses and splits all samples across the train/valid/test parsers.

        Parameters
        ----------
        stage : string, optional
            Stage of training (training, validation, testing), by default None
        """
        # here, we will actually assign
        # train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            self.train_set = ImageFolder(
                root=self.data_dir, transform=self.train_set_transformation
            )

            n_val = int(np.floor(valid_size * len(self.train_set)))
            n_train = len(self.train_set) - n_val

            self.train_set, self.val_set = random_split(
                self.train_set, [n_train, n_val]
            )

        if stage == "test" or stage is None:

            self.test_set = ImageFolder(
                root=self.data_dir, transform=self.test_set_transformation
            )

    def setup_val_sampler(
        self,
        n_way: int = 15,
        n_shot: int = 5,
        n_query: int = 5,
        num_eval_tasks: int = 100,
    ) -> None:
        """Sets up the validation sampler.

        Parameters
        ----------
        n_way : int, optional
            Number of classes in a task, by default 15
        n_shot : int, optional
            Number of images per class in the support set, by default 5
        n_query : int, optional
            Number of images per class in the query set, by default 5
        """
        # The sampler needs a dataset with a "get_labels" method.
        # Check the code if you have any doubt!
        self.val_set.get_labels = lambda: [
            instance[1] for instance in self.val_set
        ]
        val_sampler = TaskSampler(
            self.val_set,
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
            n_tasks=num_eval_tasks,
        )
        return val_sampler

    def setup_train_sampler(
        self,
        n_way: int = 15,
        n_shot: int = 5,
        n_query: int = 5,
        num_training_episodes: int = 400,
    ) -> None:
        """Sets up the training sampler.

        Parameters
        ----------
        n_way : int, optional
            Number of classes in a task, by default 15
        n_shot : int, optional
            Number of images per class in the support set, by default 5
        n_query : int, optional
            Number of images per class in the query set, by default 5
        """
        self.train_set.get_labels = lambda: [
            instance[1] for instance in self.train_set
        ]
        train_sampler = TaskSampler(
            self.train_set,
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
            n_tasks=num_training_episodes,
        )
        return train_sampler

    def train_fewshot_loader(self) -> DataLoader:
        """Creates the training dataloader using the training data parser.

        Returns
        -------
        DataLoader
            returns a pytorch DataLoader class
        """
        train_sampler = self.setup_train_sampler()
        return DataLoader(
            self.train_set,
            batch_sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=train_sampler.episodic_collate_fn,
        )

    def val_fewshot_loader(self) -> DataLoader:
        """Creates the training dataloader using the training data parser.

        Returns
        -------
        DataLoader
            returns a pytorch DataLoader class
        """
        val_sampler = self.setup_val_sampler()
        return DataLoader(
            self.val_set,
            batch_sampler=val_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=val_sampler.episodic_collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        """Creates the training dataloader using the training data parser.

        Returns
        -------
        DataLoader
            returns a pytorch DataLoader class
        """
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            batch_sampler=None,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=None,
        )

    def val_dataloader(self) -> DataLoader:
        """Creates the validation dataloader using the validation data parser.

        Returns
        -------
        DataLoader
            returns a pytorch DataLoader class
        """
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            batch_sampler=None,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=None,
        )

    def test_dataloader(self) -> DataLoader:
        """Creates the testing dataloader using the testing data parser.

        Returns
        -------
        DataLoader
            returns a pytorch DataLoader class
        """
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            batch_sampler=None,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=None,
        )


# import matplotlib as plt
if __name__ == "__main__":
    # tests the dataloader module
    args = {"batch_size": 8, "image_size": 200}
    office31_loader = Office31Loader("./examples/data/amazon/images", args)
    office31_loader.setup(stage="fit")
    # i = iter(office31_loader.train_set.dataset)
    # img, label = next(i)
    # trans = transforms.ToPILImage()
    # plt.imshow(trans(img))
    # plt.savefig('test.png')
    # print(office31_loader)
