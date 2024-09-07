from pathlib import Path
import albumentations as A
import pandas as pd
from anomalib.data.visa import make_mvtec_dataset
from anomalib.data.utils import Split, InputNormalizationMethod
from pandas import DataFrame

from datamodules.base.datamodule import SSNDataModule, BgMask
from datamodules.base.dataset import SSNDataset


class VisaDataset(SSNDataset):
    """
    Dataset class for VisA dataset

    Args:
        root (Path): path to root of dataset
        category (str): one of 12 categories
        transform (A.Compose): transforms used for preprocessing
        split (Split): either train or test split
        debug (bool): debug flag for some debug printing
    """

    def __init__(
        self,
        root: Path,
        category: str,
        transform: A.Compose,
        split: Split,
        normal_flips: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(
            transform=transform,
            root=root,
            split=split,
            flips=False,
            normal_flips=normal_flips,
            supervised=False,
            debug=debug,
        )
        self.root_category = Path(root) / Path(category)

    def make_dataset(self) -> tuple[DataFrame, DataFrame]:
        samples = make_mvtec_dataset(
            self.root_category, self.split, extensions=(".png", ".jpg", ".JPG")
        )
        # return empty for anomalous
        return samples, pd.DataFrame()


class Visa(SSNDataModule):
    """
    Datamodule for VisA

    Args:
        root (Path): path to root of dataset
        category (str): one of 15 categories
        image_size ( int | tuple[int, int] | None): image size in format of (h, w)
        normalization (str | InputNormalizationMethod): normalization method for images, defaults to imagenet
        train_batch_size (int): batch size used in training
        eval_batch_size (int): batch size used in test / inference
        num_workers (int): number of dataloader workers. Must be <= 1 for supervised
        seed (int | None): seed
        debug (bool): debug flag for some debug printing
    """

    def __init__(
        self,
        root: Path | str,
        category: str,
        image_size: tuple[int, int] | None = None,
        normalization: str
        | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        seed: int | None = None,
        normal_flips: bool = False,
        debug: bool = False,
    ) -> None:
        print(f"Resolution set to: {image_size}")

        super().__init__(
            root=root,
            supervised=False,
            image_size=image_size,
            normalization=normalization,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            seed=seed,
            flips=False,
        )

        # split data is inside "visa_pytorch"
        self.split_root = self.root / "visa_pytorch"

        self.train_data = VisaDataset(
            category=category,
            transform=self.transform_train,
            split=Split.TRAIN,
            root=self.split_root,
            debug=debug,
            normal_flips=normal_flips,
        )
        self.test_data = VisaDataset(
            category=category,
            transform=self.transform_eval,
            split=Split.TEST,
            root=self.split_root,
            debug=debug,
        )
