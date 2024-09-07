import pickle
from enum import Enum
from pathlib import Path

import albumentations as A
from anomalib.data.utils import Split, LabelName, InputNormalizationMethod
from pandas import DataFrame

from datamodules.base.datamodule import SSNDataModule
from datamodules.base.dataset import SSNDataset


# 836 defect-free and 153 defective examples of hard capsules
# 846 defect-free and 345 defective examples of soft-shelled capsules
class RatioSegmented(float, Enum):
    M0 = 0
    M10 = 0.1
    M20 = 0.2
    M30 = 0.3
    M40 = 0.4
    M50 = 0.5
    M60 = 0.6
    M70 = 0.7
    M80 = 0.8
    M90 = 0.9
    M100 = 1


class FixedFoldNumber(Enum):
    F1 = 0
    F2 = 1
    F3 = 2


class Category(Enum):
    Softgel = "softgel"
    Capsule = "capsule"


def get_default_resolution(category: Category):
    if category == Category.Softgel:
        return 144, 144
    elif category == Category.Capsule:
        return 320, 192
    else:
        raise TypeError("Use sensum.Category enum")


def read_split(
    root: Path,
    category: Category,
    fold: FixedFoldNumber,
    split: Split,
    ratio_segmented: RatioSegmented,
) -> tuple[list, list]:
    fn = root / "sensum_splits" / category.value / str(fold.value) / split.value
    with open(fn / "neg.pckl", "rb") as f:
        neg_samples = pickle.load(f)
    if split == Split.TEST:
        pos_name = "pos.pckl"
    else:
        pos_name = f"pos_{int(ratio_segmented.value * 100)}.pckl"
    with open(fn / pos_name, "rb") as f:
        pos_samples = pickle.load(f)

    return neg_samples, pos_samples


class SensumDataset(SSNDataset):
    """
    Dataset class for SensumSODF dataset

    Args:
        root (Path): path to root of dataset
        category (Category): one of the two categories of medicine
        supervised (bool): flag to signal if dataset is in supervised config
        transform (A.Compose): transforms used for preprocessing
        split (Split): either train or test split
        flips (bool): flag if dataset is extended by flipping (vert, horiz, 180).
        fold (FixedFoldNumber): fold ID of 3-fold cross validation
        ratio_segmented (RatioSegmented): number of segmented images in dataset
        debug (bool): debug flag for some debug printing
    """

    def __init__(
        self,
        root: Path,
        category: Category,
        supervised: bool,
        transform: A.Compose,
        split: Split,
        flips: bool,
        normal_flips: bool,
        fold: FixedFoldNumber,
        ratio_segmented: RatioSegmented,
        debug: bool = False,
    ) -> None:
        super().__init__(
            transform=transform,
            root=root,
            split=split,
            flips=flips,
            normal_flips=normal_flips,
            supervised=supervised,
            debug=debug,
        )
        self.category = category
        self.fold = fold
        self.ratio_segmented = ratio_segmented

    def make_dataset(
        self,
    ) -> tuple[DataFrame, DataFrame]:
        # read the split with given number of segmented samples
        neg, pos = read_split(
            root=self.root,
            category=self.category,
            fold=self.fold,
            split=self.split,
            ratio_segmented=self.ratio_segmented,
        )

        # read into form "root, sample_id, split, image_path, mask_path, label_index"
        normal_samples = [
            [
                str(self.root),
                sample_id,
                self.split.value,
                str(
                    self.root
                    / self.category.value
                    / "negative/data"
                    / f"{sample_id:03}.png"
                ),
                "",
                LabelName.NORMAL,
            ]
            for sample_id, is_segmented in neg
        ]
        normal_samples = DataFrame(
            normal_samples,
            columns=[
                "path",
                "sample_id",
                "split",
                "image_path",
                "mask_path",
                "label_index",
            ],
        )

        # read into form "root, sample_id, split, image_path, mask_path, label_index" only if segmented
        anomalous_samples = [
            [
                str(self.root),
                sample_id,
                self.split.value,
                str(
                    self.root
                    / self.category.value
                    / "positive/data"
                    / f"{sample_id:03}.png"
                ),
                str(
                    self.root
                    / self.category.value
                    / "positive/annotation"
                    / f"{sample_id:03}.png"
                ),
                LabelName.ABNORMAL,
            ]
            for sample_id, is_segmented in pos
            if is_segmented
        ]
        anomalous_samples = DataFrame(
            anomalous_samples,
            columns=[
                "path",
                "sample_id",
                "split",
                "image_path",
                "mask_path",
                "label_index",
            ],
        )

        return normal_samples, anomalous_samples


class Sensum(SSNDataModule):
    """
    Datamodule for SensumSODF

    Args:
        root (Path): path to root of dataset
        image_size ( int | tuple[int, int] | None): image size in format of (h, w)
        fold (FixedFoldNumber): fold ID of 3-fold cross validation
        normalization (str | InputNormalizationMethod): normalization method for images, defaults to imagenet
        train_batch_size (int): batch size used in training
        eval_batch_size (int): batch size used in test / inference
        num_workers (int): number of dataloader workers. Must be <= 1 for supervised
        seed (int | None): seed
        flips (bool): flag if dataset is extended by flipping (vert, horiz, 180).
        ratio_segmented (RatioSegmented): number of segmented images in dataset
        debug (bool): debug flag for some debug printing
    """

    def __init__(
        self,
        root: Path | str,
        category: Category,
        fold: FixedFoldNumber,
        image_size: tuple[int, int] | None = None,
        normalization: str
        | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        seed: int | None = None,
        flips: bool = False,
        normal_flips: bool = False,
        ratio_segmented: RatioSegmented = RatioSegmented.M0,
        debug: bool = False,
    ) -> None:
        supervised = ratio_segmented != ratio_segmented.M0

        print(f"Resolution set to: {image_size}")

        super().__init__(
            root=root,
            supervised=supervised,
            image_size=image_size,
            normalization=normalization,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            seed=seed,
            flips=flips,
        )

        self.train_data = SensumDataset(
            category=category,
            fold=fold,
            transform=self.transform_train,
            split=Split.TRAIN,
            root=root,
            ratio_segmented=ratio_segmented,
            flips=flips,
            normal_flips=normal_flips,
            supervised=supervised,
            debug=debug,
        )
        self.test_data = SensumDataset(
            category=category,
            fold=fold,
            transform=self.transform_eval,
            split=Split.TEST,
            root=root,
            ratio_segmented=ratio_segmented,
            flips=flips,
            normal_flips=False,
            supervised=supervised,
            debug=debug,
        )
