from abc import ABC
from enum import Enum
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from anomalib.data.base.datamodule import collate_fn
from anomalib.data.utils import InputNormalizationMethod
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader

from datamodules.base import BgMask
from datamodules.base.dataset import SSNDataset


class SSNDataModule(LightningDataModule, ABC):
    """
    Datamodule, modified version of AnomalibDataModule used for datasets that can also be used for supervised learning

    Args:
        root (Path): path to root of dataset
        supervised (bool): flag to signal if dataset is in supervised config
        image_size ( int | tuple[int, int] | None): image size in format of (h, w)
        normalization (str | InputNormalizationMethod): normalization method for images, defaults to imagenet
        train_batch_size (int): batch size used in training
        eval_batch_size (int): batch size used in test / inference
        num_workers (int): number of dataloader workers. Must be <= 1 for supervised
        seed (int | None): seed
        flips (bool): flag if dataset is extended by flipping (vert, horiz, 180)
        mask_bg (BgMask): flag if we use background masks
    """

    def __init__(
        self,
        root: Path | str,
        supervised: bool,
        image_size: tuple[int, int] | None = None,
        normalization: str
        | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 0,
        seed: int | None = None,
        flips: bool = False,
        mask_bg: BgMask = BgMask.NONE,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.root = Path(root)
        self.supervised = supervised

        self.flips = flips
        self.mask_bg = mask_bg

        self.train_data: SSNDataset
        self.test_data: SSNDataset

        if supervised and (num_workers > 1):
            raise Exception("Can't use more workers than 1 with positive samples")

        base_transforms = [
            A.Resize(height=image_size[0], width=image_size[1], always_apply=True)
        ]

        # add normalize transform
        if normalization == InputNormalizationMethod.IMAGENET:
            base_transforms.append(
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            )
        elif normalization == InputNormalizationMethod.NONE:
            base_transforms.append(A.ToFloat(max_value=255))
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")

        base_transforms.append(ToTensorV2())

        self.transform_train = A.Compose(
            base_transforms,
            additional_targets={"bg_mask": "mask"},
            is_check_shapes=(mask_bg == BgMask.NONE),
        )

        self.transform_eval = A.Compose(
            base_transforms,
            additional_targets={"bg_mask": "mask"},
            is_check_shapes=(mask_bg == BgMask.NONE),
        )

    @property
    def is_setup(self) -> bool:
        """Checks if setup() has been called.

        At least one of [train_data, val_data, test_data] should be setup.
        """
        _is_setup: bool = False
        for data in ("train_data", "val_data", "test_data"):
            if hasattr(self, data):
                if getattr(self, data).is_setup:
                    _is_setup = True

        return _is_setup

    def setup(self, stage: str | None = None) -> None:
        """Setup train, validation and test data.

        Args:
          stage: str | None:  Train/Val/Test stages. (Default value = None)
        """
        if not self.is_setup:
            self._setup(stage)
        assert self.is_setup

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note: The stage argument is not used here. This is because, for a given instance of an AnomalibDataModule
        subclass, all three subsets are created at the first call of setup(). This is to accommodate the subset
        splitting behaviour of anomaly tasks, where the validation set is usually extracted from the test set, and
        the test set must therefore be created as early as the `fit` stage.
        """
        assert self.train_data is not None
        assert self.test_data is not None

        self.train_data.setup()
        self.test_data.setup()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader."""
        if self.num_workers > 0:
            persist = True
        else:
            persist = False

        return DataLoader(
            dataset=self.train_data,
            shuffle=True,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            persistent_workers=persist,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(
            dataset=self.test_data,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
