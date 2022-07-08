import typing
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from image_classification_simulation.data.data_loader import MyDataModule
from image_classification_simulation.data.fsl_sampler import TaskSampler
from transformers import ViTFeatureExtractor


class Office31Loader(MyDataModule):  # pragma: no cover
    """Data module class.

    Prepares dataset parsers and instantiates data loaders.
    """

    def validate_hparams(
        self, hyper_params: typing.Dict[typing.AnyStr, typing.Any]
    ) -> None:
        """Validates the hyper-parameters.

        Parameters
        ----------
        hyper_params : typing.Dict[typing.AnyStr, typing.Any]
            Hyper-parameters relevant to the dataloader module.
        """
        if "num_workers" in hyper_params:
            self.num_workers = hyper_params["num_workers"]
        else:
            self.num_workers = 1
            print("Number of workers set to:", self.num_workers)
        if "train_test_split" in hyper_params:
            self.train_test_split = hyper_params["train_test_split"]
        else:
            self.train_test_split = 0.15
        if "image_size" in hyper_params:
            self.image_size = hyper_params["image_size"]
        else:
            self.image_size = 224
            print("image size set to:", self.image_size)

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
        self.validate_hparams(hyper_params)

        self.train_set_transformation = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

        self.test_set_transformation = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

        self.dataset = ImageFolder(
            root=self.data_dir, transform=self.train_set_transformation
        )

        # get number of class from ImageFolder object
        self.num_classes = len(self.dataset.classes)
        hyper_params["num_classes"] = self.num_classes

    def setup(self, stage: str = None):
        """Parses and splits all samples across the train/valid/test parsers.

        Parameters
        ----------
        stage : string, optional
            Stage of training (training, validation, testing), by default None
        valid_size : float, optional
            Fraction of the dataset to be used for validation, by default 0.1
        test_size : float, optional
            Fraction of the dataset to be used for testing, by default 0.1
        """
        n_val = int(np.floor(self.train_test_split * len(self.dataset)))
        n_test = int(np.floor(self.train_test_split * len(self.dataset)))

        n_train = len(self.dataset) - n_val - n_test

        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset, [n_train, n_val, n_test]
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


class Office31FewshotLoader(Office31Loader):
    """Few shot Officee31 data loader class."""

    def validate_hparams(
        self, hyper_params: typing.Dict[typing.AnyStr, typing.Any]
    ) -> None:
        """Validates the hyper-parameters.

        Parameters
        ----------
        hyper_params : typing.Dict[typing.AnyStr, typing.Any]
            Hyper-parameters relevant to the dataloader module.
        """
        super().validate_hparams(hyper_params)
        if "n_way" in hyper_params:
            self.n_way = hyper_params["n_way"]
        else:
            self.n_way = 15
        if "n_shot" in hyper_params:
            self.n_shot = hyper_params["n_shot"]
        else:
            self.n_shot = 5
        if "n_query" in hyper_params:
            self.n_query = hyper_params["n_query"]
        else:
            self.n_query = 5
        if "num_training_episodes" in hyper_params:
            self.num_training_episodes = hyper_params["num_training_episodes"]
        else:
            self.num_training_episodes = 400
        if "num_eval_tasks" in hyper_params:
            self.num_eval_tasks = hyper_params["num_eval_tasks"]
        else:
            self.num_eval_tasks = 100

    def setup(
        self,
        stage: str = "fit",
        valid_size: float = 0.1,
        test_size: float = 0.1,
    ):
        """Parses and splits all samples across the train/valid/test parsers.

        Parameters
        ----------
        stage : string, optional
            Stage of training (training, validation, testing), by default None
        valid_size : float, optional
            Fraction of the dataset to be used for validation, by default 0.1
        test_size : float, optional
            Fraction of the dataset to be used for testing, by default 0.1
        """
        self.dataset = ImageFolder(
            root=self.data_dir, transform=self.train_set_transformation
        )

        n_val = int(np.floor(valid_size * len(self.dataset)))
        n_test = int(np.floor(test_size * len(self.dataset)))

        n_train = len(self.dataset) - n_val - n_test

        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset, [n_train, n_val, n_test]
        )

        self.train_set.get_labels = lambda: [
            instance[1] for instance in self.train_set
        ]

        self.val_set.get_labels = lambda: [
            instance[1] for instance in self.val_set
        ]

        self.train_sampler = TaskSampler(
            self.train_set,
            n_way=self.n_way,
            n_shot=self.n_shot,
            n_query=self.n_query,
            n_tasks=self.num_training_episodes,
        )

        self.eval_sampler = TaskSampler(
            self.val_set,
            n_way=self.n_way,
            n_shot=self.n_shot,
            n_query=self.n_query,
            n_tasks=self.num_eval_tasks,
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
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_sampler.episodic_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Creates the training dataloader using the training data parser.

        Returns
        -------
        DataLoader
            returns a pytorch DataLoader class
        """
        return DataLoader(
            self.val_set,
            batch_sampler=self.eval_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.eval_sampler.episodic_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Creates the training dataloader using the training data parser.

        Returns
        -------
        DataLoader
            returns a pytorch DataLoader class
        """
        return DataLoader(
            self.test_set,
            batch_sampler=self.eval_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.eval_sampler.episodic_collate_fn,
        )


class Office31LoaderViT(Office31Loader):  # pragma: no cover
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
        self.validate_hparams(hyper_params)

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

        self.train_set_transformation = transforms.Compose(
            [
                transforms.Resize(self.feature_extractor.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.feature_extractor.image_mean,
                    std=self.feature_extractor.image_std,
                ),
            ]
        )

        self.val_set_transformation = transforms.Compose(
            [
                transforms.Resize(self.feature_extractor.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.feature_extractor.image_mean,
                    std=self.feature_extractor.image_std,
                ),
            ]
        )


# import matplotlib as plt
if __name__ == "__main__":
    # tests the dataloader module
    args = {"batch_size": 8}
    office31_loader = Office31LoaderViT(
        "./examples/data/domain_adaptation_images/amazon/images", args
    )
    office31_loader.setup(stage="fit")
    i = iter(office31_loader.train_set.dataset)
    img, label = next(i)
    trans = transforms.ToPILImage()
    plt.imshow(trans(img))
    # plt.savefig('test.png')
    print(office31_loader)

    hparams = {
        "num_workers": 2,
        "batch_size": 32,
        "n_way": 31,
        "n_shot": 10,
        "n_query": 10,
        "num_training_episodes": 400,
        "num_eval_tasks": 50,
    }
    office_loader = Office31FewshotLoader(
        data_dir="./examples/data/domain_adaptation_images/amazon/images/",
        hyper_params=hparams,
    )
    office_loader.setup(None, 0.1, 0.1)
    train_loader = office_loader.train_dataloader()
    val_loader = office_loader.val_dataloader()
    test_loader = office_loader.test_dataloader()

    enumerate(train_loader)
    enumerate(val_loader)
    enumerate(test_loader)
