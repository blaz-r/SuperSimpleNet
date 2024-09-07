from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from anomalib.data.utils import Split, read_image
from pandas import DataFrame
from scipy.ndimage import distance_transform_edt
from torch import Tensor
from torch.utils.data import Dataset

from datamodules.base import BgMask


class SSNDataset(Dataset):
    """
    Dataset, modified version of AnomalibDataset used for datasets that can also be used for supervised learning

    Args:
        root (Path): path to root of dataset
        supervised (bool): flag to signal if dataset is in supervised config
        transform (A.Compose): transforms used for preprocessing
        split (Split): either train or test split
        flips (bool): flag if dataset is extended by flipping (vert, horiz, 180).
        normal_flips (bool): flag if we also flip normal data.
        debug (bool): debug flag for some debug printing
    """

    def __init__(
        self,
        root: Path,
        supervised: bool,
        transform: A.Compose,
        split: Split,
        flips: bool,
        normal_flips: bool,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.transform = transform
        self._normal_samples: DataFrame
        self._anomalous_samples: DataFrame

        self.supervised = supervised

        self.root = root
        self.split = split
        self.flips = flips
        self.normal_flips = normal_flips

        self.counter = 0
        self.generated_num_pos: int = 0
        self.neg_retrieval_freq: np.ndarray

        self.num_pos: int
        self.num_neg: int

        self.vflip = A.VerticalFlip(always_apply=True)
        self.hflip = A.HorizontalFlip(always_apply=True)
        self.rotate = A.Rotate(limit=(180, 180), always_apply=True)

        self.random_normal_rotate = A.Compose(
            [
                A.VerticalFlip(p=0.3),
                A.HorizontalFlip(p=0.3),
                A.Rotate(limit=90, p=0.3),
            ]
        )

        self.debug = debug
        if self.debug:
            print("created")

    @property
    def samples(self) -> tuple[DataFrame, DataFrame]:
        """Get the samples dataframe."""
        if not self.is_setup:
            raise RuntimeError("Dataset is not setup yet. Call setup() first.")
        return self._normal_samples, self._anomalous_samples

    @property
    def is_setup(self) -> bool:
        """Checks if setup() been called."""
        return hasattr(self, "_normal_samples") and hasattr(self, "_anomalous_samples")

    def setup(self) -> None:
        """Load data/metadata into memory."""
        if not self.is_setup:
            self._setup()
        assert self.is_setup, "setup() should set self._samples"

    def make_dataset(self) -> tuple[DataFrame, DataFrame]:
        """
        Make and return two dataframes for normal and anomalous samples

        :return: normal samples, anomalous samples
        """
        raise NotImplementedError

    def _setup(self) -> None:
        self._normal_samples, self._anomalous_samples = self.make_dataset()

        self.num_neg = len(self._normal_samples)
        self.num_pos = len(self._anomalous_samples)

        if self.supervised:
            # if have positive samples we use frequency sampling
            self.neg_retrieval_freq = np.zeros(shape=self.num_neg)

            # if we use flips, we have 4 times as much positive cases
            if self.flips:
                neg_sample_size = self.num_pos * 4
                self.generated_num_pos = self.num_pos * 4
            else:
                neg_sample_size = self.num_pos
                self.generated_num_pos = self.num_pos

            # sample with replacement if we need to sample more than we have
            if neg_sample_size > self.num_neg:
                self.replace = True
            else:
                self.replace = False
                neg_sample_size = self.num_neg

            self.neg_imgs_permutation = np.random.choice(
                range(self.num_neg),
                size=neg_sample_size,
                replace=self.replace,
            )

    def __len__(self) -> int:
        """Get length of the dataset."""
        if self.split == Split.TEST:
            # in test, we return true len
            return self.num_pos + self.num_neg
        elif self.supervised:
            # we have positive, return size of balanced data
            if self.flips:
                # flip v, h, 180 - so we get 4x the pos, balanced with 4times the neg
                return self.num_pos * 8
            else:
                # if no flipping augmentation
                return self.num_pos * 2
        else:
            # if we don't have anomalous return num of neg
            return self.num_neg

    def generate_permutation(self) -> None:
        """
        Generate negative images permutation by inverse frequency sampling

        """
        self.counter = 0
        sample_probability = 1 - (
            self.neg_retrieval_freq / np.max(self.neg_retrieval_freq)
        )
        sample_probability = sample_probability - np.median(sample_probability) + 1
        sample_probability = sample_probability ** (np.log(len(sample_probability)) * 4)
        sample_probability = sample_probability / np.sum(sample_probability)

        # use replace=False to get only unique values
        self.neg_imgs_permutation = np.random.choice(
            range(self.num_neg),
            size=self.generated_num_pos,
            p=sample_probability,
            replace=self.replace,
        )

    def get_sample_data(self, index) -> tuple[str, str, int]:
        """
        Get image and mask path, label_index (label)

        Args:
            index: current index of image to retrieve

        Returns:
            (tuple[str, str, int]): image path, mask path, label index
        """
        if self.split == Split.TRAIN:
            if index >= self.generated_num_pos:
                if self.supervised:
                    permutation_index = index % self.generated_num_pos
                    ix = self.neg_imgs_permutation[permutation_index]
                    self.neg_retrieval_freq[ix] = self.neg_retrieval_freq[ix] + 1
                else:
                    ix = index
                image_path = self._normal_samples.iloc[ix].image_path
                mask_path = self._normal_samples.iloc[ix].mask_path
                label_index = self._normal_samples.iloc[ix].label_index
            else:
                # to get actual index of positive
                ix = index % self.num_pos
                image_path = self._anomalous_samples.iloc[ix].image_path
                mask_path = self._anomalous_samples.iloc[ix].mask_path
                label_index = self._anomalous_samples.iloc[ix].label_index
        # test
        else:
            if index < self.num_neg:
                ix = index
                image_path = self._normal_samples.iloc[ix].image_path
                mask_path = self._normal_samples.iloc[ix].mask_path
                label_index = self._normal_samples.iloc[ix].label_index
            else:
                ix = index - self.num_neg
                image_path = self._anomalous_samples.iloc[ix].image_path
                mask_path = self._anomalous_samples.iloc[ix].mask_path
                label_index = self._anomalous_samples.iloc[ix].label_index

        return image_path, mask_path, label_index

    def get_flip_augmentation(self, index) -> A.DualTransform | None | A.Compose:
        """
        Get specified flip augmentation for current image if flips is true.

        According to index:
        0..n = none, n..2n = vflip, 2n..3n = hflip, 3n..4n=rot

        Args:
            index:

        Returns:

        """
        if self.split == Split.TRAIN:
            # index indicates that current sample is anomalous
            if (index < self.generated_num_pos) and self.flips:
                # assign augmentation according to index
                # 0..n = none, n..2n = vflip, 2n..3n = hflip, 3n..4n=rot
                aug_idx = index // self.num_pos
                if aug_idx == 0:
                    return None
                elif aug_idx == 1:
                    return self.vflip
                elif aug_idx == 2:
                    return self.hflip
                elif aug_idx == 3:
                    return self.rotate
                else:
                    raise Exception("Shouldn't happen")
            # normal sample and normflisp is true
            elif self.normal_flips:
                return self.random_normal_rotate
            else:
                return None

    def distance_transform(
        self, mask: np.ndarray, max_val: float, p: float
    ) -> np.ndarray:
        h, w = mask.shape[:2]
        dst_trf = np.zeros((h, w))

        num_labels, labels = cv2.connectedComponents(
            (mask * 255.0).astype(np.uint8), connectivity=8
        )
        for idx in range(1, num_labels):
            mask_roi = np.zeros((h, w))
            k = labels == idx
            mask_roi[k] = 255
            dst_trf_roi = distance_transform_edt(mask_roi)
            if dst_trf_roi.max() > 0:
                dst_trf_roi = dst_trf_roi / dst_trf_roi.max()
                dst_trf_roi = (dst_trf_roi**p) * max_val
            dst_trf += dst_trf_roi

        dst_trf[mask == 0] = 1
        return np.array(dst_trf, dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            Union[dict[str, Tensor], dict[str, str | Tensor]]: Dict of image tensor during training.
                Otherwise, Dict containing image path, target path, image tensor, label and transformed bounding box.
        """
        if (
            (self.split == Split.TRAIN)
            and self.supervised
            and (self.counter >= len(self))
        ):
            # we have labeled, so we need freq distr
            self.generate_permutation()

        image_path, mask_path, label_index = self.get_sample_data(index)

        image = read_image(image_path)
        item = dict(image_path=image_path, label=label_index)

        if label_index == 0:
            # normal are all zero
            mask = np.zeros(shape=image.shape[:2])
        else:
            mask = cv2.imread(mask_path, flags=0) / 255.0

        if (self.flips or self.normal_flips) and (self.split == Split.TRAIN):
            # if current image is selected to be flip augmented
            flip_augmentation = self.get_flip_augmentation(index)
            if flip_augmentation is not None:
                flip_transformed = flip_augmentation(image=image, mask=mask)
                image = flip_transformed["image"]
                mask = flip_transformed["mask"]

        transformed = self.transform(image=image, mask=mask)

        item["image"] = transformed["image"]
        item["mask_path"] = mask_path
        item["mask"] = transformed["mask"]

        self.counter = self.counter + 1

        if self.debug and self.counter == len(self):
            # print number of elements in freq table at end of epoch
            print(self.neg_retrieval_freq.sum())

        return item
