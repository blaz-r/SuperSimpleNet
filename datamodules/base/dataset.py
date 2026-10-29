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

from datamodules.base import Supervision


class SSNDataset(Dataset):
    """
    Dataset, modified version of AnomalibDataset used for datasets that can also be used for supervised learning

    Args:
        root (Path): path to root of dataset
        supervision (Supervision): flag to signal if dataset is in supervised config
        transform (A.Compose): transforms used for preprocessing
        split (Split): either train or test split
        flips (bool): flag if dataset is extended by flipping (vert, horiz, 180).
        normal_flips (bool): flag if we also flip normal data.
        dilate (int|None) if an int = size of dilation square, if None - not applied (default None)
        dt (tuple[int, int] | None) distance transform params (w, p), if None - not applied (default None)
        debug (bool): debug flag for some debug printing
    """

    def __init__(
        self,
        root: Path,
        supervision: Supervision,
        transform: A.Compose,
        split: Split,
        flips: bool,
        normal_flips: bool,
        dilate: int | None = None,
        dt: tuple[int, int] | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.transform = transform
        self._normal_samples: DataFrame
        self._anomalous_samples: DataFrame

        self.supervision = supervision

        self.root = root
        self.split = split
        self.flips = flips
        self.normal_flips = normal_flips

        self.dilate = dilate
        self.dt = dt

        self.counter = 0
        self.generated_num_pos: int = 0
        self.neg_retrieval_freq: np.ndarray

        self.num_pos: int
        self.num_neg: int

        self.vflip = A.VerticalFlip(p=1.0)
        self.hflip = A.HorizontalFlip(p=1.0)
        self.rotate = A.Rotate(limit=(180, 180), border_mode=cv2.BORDER_REFLECT_101, p=1.0)

        self.random_normal_rotate = A.Compose(
            [
                A.VerticalFlip(p=0.3),
                A.HorizontalFlip(p=0.3),
                A.Rotate(limit=90, border_mode=cv2.BORDER_REFLECT_101, p=0.3),
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
        self.counter = 0

        self._normal_samples, self._anomalous_samples = self.make_dataset()

        self.num_neg = len(self._normal_samples)
        self.num_pos = len(self._anomalous_samples)

        if self.supervision != Supervision.UNSUPERVISED:
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
        if self.split in [Split.TEST, Split.VAL]:
            # in test, we return true len
            return self.num_pos + self.num_neg
        elif self.supervision == Supervision.UNSUPERVISED:
            # if we don't have anomalous return num of neg
            return self.num_neg
        else:
            # we have positive, return size of balanced data
            if self.flips:
                # flip v, h, 180 - so we get 4x the pos, balanced with 4times the neg
                return self.num_pos * 8
            else:
                # if no flipping augmentation
                return self.num_pos * 2

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

    def get_sample_data(self, index) -> tuple[str, str, int, int]:
        """
        Get image and mask path, label_index (label)

        Args:
            index: current index of image to retrieve

        Returns:
            (tuple[str, str, int]): image path, mask path, label index
        """
        if self.split == Split.TRAIN:
            if index >= self.generated_num_pos:
                if self.supervision == Supervision.UNSUPERVISED:
                    ix = index
                else:
                    permutation_index = index % self.generated_num_pos
                    ix = self.neg_imgs_permutation[permutation_index]
                    self.neg_retrieval_freq[ix] = self.neg_retrieval_freq[ix] + 1
                image_path = self._normal_samples.iloc[ix].image_path
                mask_path = self._normal_samples.iloc[ix].mask_path
                label_index = self._normal_samples.iloc[ix].label_index
                is_segmented = self._normal_samples.iloc[ix].get("is_segmented", True)
            else:
                # to get actual index of positive
                ix = index % self.num_pos
                image_path = self._anomalous_samples.iloc[ix].image_path
                mask_path = self._anomalous_samples.iloc[ix].mask_path
                label_index = self._anomalous_samples.iloc[ix].label_index
                is_segmented = self._anomalous_samples.iloc[ix].get(
                    "is_segmented", True
                )
        # test
        else:
            if index < self.num_neg:
                ix = index
                image_path = self._normal_samples.iloc[ix].image_path
                mask_path = self._normal_samples.iloc[ix].mask_path
                label_index = self._normal_samples.iloc[ix].label_index
                is_segmented = self._normal_samples.iloc[ix].get("is_segmented", True)

            else:
                ix = index - self.num_neg
                image_path = self._anomalous_samples.iloc[ix].image_path
                mask_path = self._anomalous_samples.iloc[ix].mask_path
                label_index = self._anomalous_samples.iloc[ix].label_index
                is_segmented = self._anomalous_samples.iloc[ix].get(
                    "is_segmented", True
                )

        return image_path, mask_path, label_index, is_segmented

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
        """
        Apply distance transform to weight the pixels according to distance from center of shape.
        Distance is additionally transformed using linear function: Omega(d) = w * d^p
        where w is max_val

        From:
        https://github.com/vicoslab/mixed-segdec-net-comind2021/blob/
        21583eb22e719a70fee388ce45f0f7f27f529926/data/dataset.py#L122

        Args:
            mask (np.ndarray): input segmentation GT mask
            max_val (float): scalar weight for pos. pixels (w in eq.)
            p (float): rate of decreasing the pixel importance
        """
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
            and self.supervision != Supervision.UNSUPERVISED
            and (self.counter >= len(self))
        ):
            # we have labeled, so we need freq distr
            self.generate_permutation()

        image_path, mask_path, label_index, is_segmented = self.get_sample_data(index)

        image = read_image(image_path)
        item = dict(
            image_path=image_path,
            label=label_index,
            is_segmented=is_segmented,
        )

        if label_index == 0 or not is_segmented:
            # normal or not segmented are all zero
            mask = np.zeros(shape=image.shape[:2])
        else:
            mask = cv2.imread(mask_path, flags=0) / 255.0

            if self.dilate is not None and self.split == Split.TRAIN:
                mask = cv2.dilate(mask, np.ones((self.dilate, self.dilate)))

        if (self.flips or self.normal_flips) and (self.split == Split.TRAIN):
            # if current image is selected to be flip augmented
            flip_augmentation = self.get_flip_augmentation(index)
            if flip_augmentation is not None:
                flip_transformed = flip_augmentation(image=image, mask=mask)
                image = flip_transformed["image"]
                mask = flip_transformed["mask"]

        if self.dt is not None:
            # apply distance transform
            wp, p = self.dt
            # if normal all 1, otherwise dist transform
            if label_index == 0:
                loss_mask = np.ones(shape=image.shape[:2])
            else:
                loss_mask = self.distance_transform(mask, wp, p)

            transformed = self.transform(image=image, mask=mask, loss_mask=loss_mask)
            item["loss_mask"] = transformed["loss_mask"]
        else:
            transformed = self.transform(image=image, mask=mask)

        item["image"] = transformed["image"]
        item["mask_path"] = mask_path
        item["mask"] = transformed["mask"]

        self.counter = self.counter + 1

        if self.debug and self.counter == len(self):
            # print number of elements in freq table at end of epoch
            print(self.neg_retrieval_freq.sum())

        return item
