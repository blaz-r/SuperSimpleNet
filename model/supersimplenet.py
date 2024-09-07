import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, LRScheduler
from torchvision.transforms import GaussianBlur

from common.perlin_noise import rand_perlin_2d
from .feature_extractor import FeatureExtractor


class SuperSimpleNet(nn.Module):
    """
    SuperSimpleNet model

    Args:
        image_size: tuple with image dims (h, w)
        config: dict with model properties
    """

    def __init__(self, image_size: tuple[int, int], config):
        super().__init__()
        self.image_size = image_size
        self.config = config
        self.feature_extractor = FeatureExtractor(
            backbone=config.get("backbone", "wide_resnet50_2"),
            layers=config.get("layers", ["layer2", "layer3"]),
            patch_size=config.get("patch_size", 3),
            image_size=image_size,
        )
        # feature channels, height and width
        fc, fh, fw = self.feature_extractor.feature_dim
        self.fh = fh
        self.fw = fw
        self.feature_adaptor = FeatureAdaptor(projection_dim=fc)
        self.discriminator = Discriminator(
            projection_dim=fc,
            hidden_dim=1024,
            feature_w=fw,
            feature_h=fh,
            config=config,
        )

        self.anomaly_generator = AnomalyGenerator(
            noise_mean=0,
            noise_std=config.get("noise_std", 0.015),
            feature_w=fw,
            feature_h=fh,
            f_dim=fc,
            config=config,
        )
        self.anomaly_map_generator = AnomalyMapGenerator(
            output_size=image_size, sigma=4
        )

    def forward(
        self,
        images: Tensor,
        mask: Tensor = None,
        label: Tensor = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        # feature extraction, upscaling and neigh. aggregation
        # [B, F_dim, H, W]
        features = self.feature_extractor(images)
        adapted = self.feature_adaptor(features)

        if self.training:
            # add noise to features
            if self.config["noise"]:
                # also returns adjusted labels and masks
                final_features, mask, label = self.anomaly_generator(
                    adapted, mask, label
                )
            else:
                final_features = adapted

            anomaly_map, anomaly_score = self.discriminator(final_features)
            return anomaly_map, anomaly_score, mask, label
        else:
            anomaly_map, anomaly_score = self.discriminator(adapted)
            anomaly_map = self.anomaly_map_generator(anomaly_map)

            return anomaly_map, anomaly_score

    def get_optimizers(self) -> tuple[Optimizer, LRScheduler]:
        seg_params, dec_params = self.discriminator.get_params()
        optim = AdamW(
            [
                {
                    "params": self.feature_adaptor.parameters(),
                    "lr": self.config["adapt_lr"],
                },
                {
                    "params": seg_params,
                    "lr": self.config["seg_lr"],
                    "weight_decay": 0.00001,
                },
                {
                    "params": dec_params,
                    "lr": self.config["dec_lr"],
                    "weight_decay": 0.00001,
                },
            ]
        )
        sched = MultiStepLR(
            optim,
            milestones=[self.config["epochs"] * 0.8, self.config["epochs"] * 0.9],
            gamma=self.config["gamma"],
        )

        return optim, sched

    def save_model(self, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        state_dict = self.state_dict()
        # exclude feat extractor since it's pretrained
        saving_state_dict = OrderedDict(
            {
                n: k
                for n, k in state_dict.items()
                if not n.startswith("feature_extractor")
            }
        )

        torch.save(saving_state_dict, path / "weights.pt")

    def load_model(self, path):
        print(f"Loading model: {path}")
        self.load_state_dict(torch.load(path), strict=False)


def init_weights(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)


class FeatureAdaptor(nn.Module):
    def __init__(self, projection_dim: int):
        super().__init__()
        # linear layer equivalent
        self.projection = nn.Conv2d(
            in_channels=projection_dim,
            out_channels=projection_dim,
            kernel_size=1,
            stride=1,
        )
        self.apply(init_weights)

    def forward(self, features: Tensor) -> Tensor:
        return self.projection(features)


def _conv_block(in_chanels, out_chanels, kernel_size, padding="same"):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_chanels,
            out_channels=out_chanels,
            kernel_size=kernel_size,
            padding=padding,
        ),
        nn.BatchNorm2d(out_chanels),
        nn.ReLU(inplace=True),
    )


class Discriminator(nn.Module):
    def __init__(
        self, projection_dim: int, hidden_dim: int, feature_w, feature_h, config
    ):
        super().__init__()
        self.fw = feature_w
        self.fh = feature_h
        self.stop_grad = config.get("stop_grad", False)

        # 1x1 conv - linear layer equivalent
        self.seg = nn.Sequential(
            nn.Conv2d(
                in_channels=projection_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

        self.dec_head = _conv_block(
            in_chanels=projection_dim + 1, out_chanels=128, kernel_size=5
        )

        self.map_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.map_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.dec_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dec_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.fc_score = nn.Linear(in_features=128 * 2 + 2, out_features=1)

        self.apply(init_weights)

    def get_params(self):
        seg_params = self.seg.parameters()
        dec_params = list(self.dec_head.parameters()) + list(self.fc_score.parameters())
        return seg_params, dec_params

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        # get anomaly map from seg head
        map = self.seg(input)

        map_dec_copy = map
        if self.stop_grad:
            map_dec_copy = map_dec_copy.detach()
        # dec conv layer takes feat + map
        mask_cat = torch.cat((input, map_dec_copy), dim=1)
        dec_out = self.dec_head(mask_cat)

        dec_max = self.dec_max_pool(dec_out)
        dec_avg = self.dec_avg_pool(dec_out)

        map_max = self.map_max_pool(map)
        if self.stop_grad:
            map_max = map_max.detach()

        map_avg = self.map_avg_pool(map)
        if self.stop_grad:
            map_avg = map_avg.detach()

        # final dec layer: conv channel max and avg and map max and avg
        dec_cat = torch.cat((dec_max, dec_avg, map_max, map_avg), dim=1).squeeze(
            dim=(2, 3)
        )
        score = self.fc_score(dec_cat).squeeze(dim=1)

        return map, score


class AnomalyGenerator(nn.Module):
    def __init__(
        self,
        noise_mean: float,
        noise_std: float,
        feature_h: int,
        feature_w: int,
        f_dim: int,
        config: dict,
        perlin_range: tuple[int, int] = (0, 6),
    ):
        super().__init__()

        self.noise_mean = noise_mean
        self.noise_std = noise_std

        self.min_perlin_scale = perlin_range[0]
        self.max_perlin_scale = perlin_range[1]

        self.height = feature_h
        self.width = feature_w
        self.f_dim = f_dim

        self.config = config

        self.perlin_height = self.next_power_2(self.height)
        self.perlin_width = self.next_power_2(self.width)

    @staticmethod
    def next_power_2(num):
        return 1 << (num - 1).bit_length()

    def generate_perlin(self, batches) -> Tensor:
        """
        Generate 2d perlin noise masks with dims [b, 1, self.h, self.w]

        Args:
            batches: number of batches (different masks)

        Returns:
            tensor with b perlin binarized masks
        """
        perlin = []
        for _ in range(batches):
            perlin_scalex = (
                2
                ** (
                    torch.randint(
                        self.min_perlin_scale, self.max_perlin_scale, (1,)
                    ).numpy()[0]
                )
            )
            perlin_scaley = (
                2
                ** (
                    torch.randint(
                        self.min_perlin_scale, self.max_perlin_scale, (1,)
                    ).numpy()[0]
                )
            )

            perlin_noise = rand_perlin_2d(
                (self.perlin_height, self.perlin_width), (perlin_scalex, perlin_scaley)
            )
            # original is power of 2 scale, so fit to our size
            perlin_noise = F.interpolate(
                perlin_noise.reshape(1, 1, self.perlin_height, self.perlin_width),
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )
            threshold = self.config["perlin_thr"]
            # binarize
            perlin_thr = torch.where(perlin_noise > threshold, 1, 0)

            chance_anomaly = torch.rand(1).numpy()[0]
            if chance_anomaly > 0.5:
                if self.config["no_anomaly"] == "full":
                    # entire image is anomaly
                    perlin_thr = torch.ones_like(perlin_thr)
                elif self.config["no_anomaly"] == "empty":
                    # no anomaly
                    perlin_thr = torch.zeros_like(perlin_thr)
                # if none -> don't add

            perlin.append(perlin_thr)
        return torch.cat(perlin)

    def forward(
        self, input: Tensor, mask: Tensor, labels: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        b, _, h, w = mask.shape

        # duplicate
        input = torch.cat((input, input))
        mask = torch.cat((mask, mask))
        labels = torch.cat((labels, labels))

        noise = torch.normal(
            mean=self.noise_mean,
            std=self.noise_std,
            size=input.shape,
            device=input.device,
            requires_grad=False,
        )

        # mask indicating which regions will have noise applied
        # [B * 2, 1, H, W] initial all masked as anomalous
        noise_mask = torch.ones(
            b * 2, 1, h, w, device=input.device, requires_grad=False
        )

        if not self.config["bad"]:
            # reshape so it can be multiplied
            masking_labels = labels.reshape(b * 2, 1, 1, 1)
            # if option w/o bad, don't apply additional to bad (label=1 -> bad)
            noise_mask = noise_mask * (1 - masking_labels)

        if not self.config["overlap"]:
            # if no overlap, don't apply to already anomalous regions (mask=1 -> bad)
            noise_mask = noise_mask * (1 - mask)

        if self.config["perlin"]:
            # [B * 2, 1, H, W]
            perlin_mask = self.generate_perlin(b * 2).to(input.device)
            # if perlin only apply where perlin mask is 1
            noise_mask = noise_mask * perlin_mask
        else:
            # if not perlin, original SN strategy: don't apply to first half of samples
            noise_mask[:b, ...] = 0

        # update gt mask
        mask = mask + noise_mask
        # binarize
        mask = torch.where(mask > 0, 1, 0)

        # make new labels. 1 if any part of maks is 1, 0 otherwise
        new_anomalous = mask.reshape(b * 2, -1).any(dim=1).type(torch.float32)
        labels = labels + new_anomalous
        # binarize
        labels = torch.where(labels > 0, 1, 0)

        # apply masked noise
        perturbed = input + noise * noise_mask

        return perturbed, mask, labels


class AnomalyMapGenerator(nn.Module):
    def __init__(self, output_size: tuple[int, int], sigma: float):
        super().__init__()
        self.size = output_size
        kernel_size = 2 * math.ceil(3 * sigma) + 1
        self.blur = GaussianBlur(kernel_size=kernel_size, sigma=4)

    def forward(self, input: Tensor) -> Tensor:
        # upscale & smooth
        anomaly_map = F.interpolate(input, size=self.size, mode="bilinear")
        anomaly_map = self.blur(anomaly_map)
        return anomaly_map
