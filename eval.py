import copy
import json
from pathlib import Path

import torch
from anomalib.utils.metrics import AUROC, AUPRO
from torchmetrics import Metric, AveragePrecision
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
import pandas as pd

from common.results_writer import ResultsWriter
from common.visualizer import Visualizer
from datamodules import sensum, ksdd2
from datamodules.mvtec import MVTec
from datamodules.visa import Visa
from datamodules.ksdd2 import KSDD2
from datamodules.sensum import Sensum
from model.supersimplenet import SuperSimpleNet


@torch.no_grad()
def eval(
    model: SuperSimpleNet,
    datamodule: LightningDataModule,
    device: str,
    image_metrics: dict[str, Metric],
    pixel_metrics: dict[str, Metric],
    normalize: bool = True,
    image_save_path: Path = None,
    score_save_path: Path = None,
):
    model.to(device)
    model.eval()

    # for anomaly map max as image score
    seg_image_metrics = {}

    for m_name, metric in image_metrics.items():
        metric.cpu()
        metric.reset()

        seg_image_metrics[f"seg-{m_name}"] = copy.deepcopy(metric)

    for metric in pixel_metrics.values():
        metric.cpu()
        metric.reset()

    test_loader = datamodule.test_dataloader()
    results = {
        "anomaly_map": [],
        "gt_mask": [],
        "score": [],
        "seg_score": [],
        "label": [],
        "image_path": [],
        "mask_path": [],
    }
    for batch in tqdm(test_loader, position=0, leave=True):
        image_batch = batch["image"].to(device)
        anomaly_map, anomaly_score = model.forward(image_batch)

        anomaly_map = anomaly_map.detach().cpu()
        anomaly_score = anomaly_score.detach().cpu()

        results["anomaly_map"].append(anomaly_map.detach().cpu())
        results["gt_mask"].append(batch["mask"].detach().cpu())

        results["score"].append(torch.sigmoid(anomaly_score))
        results["seg_score"].append(
            anomaly_map.reshape(anomaly_map.shape[0], -1).max(dim=1).values
        )
        results["label"].append(batch["label"].detach().cpu())

        results["image_path"].extend(batch["image_path"])
        results["mask_path"].extend(batch["mask_path"])

    results["anomaly_map"] = torch.cat(results["anomaly_map"])
    results["score"] = torch.cat(results["score"])
    results["seg_score"] = torch.cat(results["seg_score"])
    results["gt_mask"] = torch.cat(results["gt_mask"])
    results["label"] = torch.cat(results["label"])

    # normalize
    if normalize:
        results["anomaly_map"] = (
            results["anomaly_map"] - results["anomaly_map"].flatten().min()
        ) / (
            results["anomaly_map"].flatten().max()
            - results["anomaly_map"].flatten().min()
        )
        results["score"] = (results["score"] - results["score"].min()) / (
            results["score"].max() - results["score"].min()
        )
        results["seg_score"] = (results["seg_score"] - results["seg_score"].min()) / (
            results["seg_score"].max() - results["seg_score"].min()
        )

    results_dict = {}
    for name, metric in image_metrics.items():
        metric.update(results["score"], results["label"])
        results_dict[name] = metric.to(device).compute().item()
        metric.to("cpu")

    for name, metric in seg_image_metrics.items():
        metric.update(results["seg_score"], results["label"])
        results_dict[name] = metric.to(device).compute().item()
        metric.to("cpu")

    for name, metric in pixel_metrics.items():
        try:
            # avoid nan in early stages
            am = results["anomaly_map"]
            am[am != am] = 0
            results["anomaly_map"] = am

            metric.update(results["anomaly_map"], results["gt_mask"].type(torch.float32))
            results_dict[name] = metric.to(device).compute().item()
        except RuntimeError:
            # AUPRO in some cases with early predictions crashes cuda, so just skip it in that case
            results_dict[name] = 0
        metric.to("cpu")

    for name, value in results_dict.items():
        print(f"{name}: {value} ", end="")
    print()

    if image_save_path:
        print("Visualizing")
        visualizer = Visualizer(image_save_path)
        visualizer.visualize(results)

    score_dict = {}
    if score_save_path:
        # save both segscore and score to json
        for img_path, score, seg_score, label in zip(
            results["image_path"],
            results["score"],
            results["seg_score"],
            results["label"],
        ):
            img_path = Path(img_path)

            anomaly_type = img_path.parent.name
            if anomaly_type not in score_dict:
                score_dict[anomaly_type] = {"good": {}, "bad": {}}

            # since some datasets (sensum) can have same names in bad and good
            if label == 1:
                kind = "bad"
            else:
                kind = "good"

            score_dict[anomaly_type][kind][img_path.stem] = {
                "score": score.item(),
                "seg_score": seg_score.item(),
            }

        score_save_path.mkdir(exist_ok=True, parents=True)
        with open(score_save_path / "scores.json", "w") as f:
            json.dump(score_dict, f)

    return results_dict


def get_sensum(config):
    data = []
    for category in [sensum.Category.Capsule, sensum.Category.Softgel]:
        for fold_num in range(3):
            datamodule = Sensum(
                root=Path(config["datasets_folder"]) / "SensumSODF",
                fold=sensum.FixedFoldNumber(fold_num),
                category=category,
                image_size=sensum.get_default_resolution(category),
                train_batch_size=config["batch"],
                eval_batch_size=config["batch"],
                num_workers=config["num_workers"],
                ratio_segmented=sensum.RatioSegmented.M100,
                seed=config["seed"],
                flips=False,
            )
            datamodule.setup()
            data.append((f"{category.value}_{fold_num}", datamodule))
    return data


def get_ksdd2(config):
    datamodule = KSDD2(
        root=Path(config["datasets_folder"]) / "KolektorSDD2",
        image_size=ksdd2.get_default_resolution(),
        train_batch_size=config["batch"],
        eval_batch_size=config["batch"],
        num_workers=config["num_workers"],
        num_segmented=ksdd2.NumSegmented.N246,
        seed=config["seed"],
        flips=False,
    )
    datamodule.setup()

    return [("ksdd2", datamodule)]


def get_mvtec(config):
    data = []

    categories = [
        "screw",
        "pill",
        "capsule",
        "carpet",
        "grid",
        "tile",
        "wood",
        "zipper",
        "cable",
        "toothbrush",
        "transistor",
        "metal_nut",
        "bottle",
        "hazelnut",
        "leather",
    ]

    for category in categories:
        datamodule = MVTec(
            root=Path(config["datasets_folder"]) / "mvtec",
            category=category,
            image_size=(256, 256),
            train_batch_size=config["batch"],
            eval_batch_size=config["batch"],
            num_workers=config["num_workers"],
            seed=config["seed"],
        )
        datamodule.setup()
        data.append((category, datamodule))

    return data


def get_visa(config):
    data = []

    categories = [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ]

    for category in categories:
        datamodule = Visa(
            root=Path(config["datasets_folder"]) / "visa",
            category=category,
            image_size=(256, 256),
            train_batch_size=config["batch"],
            eval_batch_size=config["batch"],
            num_workers=config["num_workers"],
            seed=config["seed"],
        )
        datamodule.setup()

        data.append((category, datamodule))

    return data


def get_avg(df):
    cat_avg = df.groupby("category").mean(numeric_only=True)
    total_avg = df.mean(axis=0, numeric_only=True).to_frame().T
    total_avg.index = ["total"]
    combined = pd.concat([cat_avg, total_avg], axis=0)

    return combined


def get_std(df):
    # take std of cat mean - this covers standard splits as well as CV for sensum
    cat_std = (
        df.groupby(["run_id", "category"])
        .mean(numeric_only=True)
        .reset_index()
        .groupby("category")
        .std()
    )

    total_std = df.groupby("run_id").mean(numeric_only=True).std(numeric_only=True)
    total_std = total_std.to_frame().T
    total_std.index = ["total"]

    combined = pd.concat([cat_std, total_std], axis=0)

    return combined


def merge_csvs(dataset, run_ids, base_path):
    # read all csv and merge into one
    joined = None
    for run_id in run_ids:
        file = base_path / str(run_id) / dataset / ("last.csv")
        print(file)
        df = pd.read_csv(file)
        if joined is None:
            joined = df
        else:
            joined = pd.concat([joined, df], axis=0)

    return joined


def get_stats(dataset, run_ids, base_path):
    joined = merge_csvs(dataset, run_ids, base_path)

    comb_avg = get_avg(joined)
    comb_std = get_std(joined)

    return comb_avg, comb_std


def generate_result_json(run_ids, datasets, res_path):
    """
    Generate json with mean and std for all passed datasets and run_ids.

    Args:
        run_ids: list of run_ids
        datasets: list of datasets
        res_path: root path of results (csvs)

    """
    res_json = {"avg": {}, "std": {}}

    for dataset in datasets:
        avg, std = get_stats(dataset, run_ids, res_path)
        avg = avg.drop(columns=["run_id"])
        std = std.drop(columns=["run_id"])

        res_json["avg"][dataset] = avg.to_dict()
        if len(run_ids) > 1:
            res_json["std"][dataset] = std.to_dict()

    Path("./res_json").mkdir(exist_ok=True, parents=True)
    with open("./res_json/ssn.json", "w") as f:
        json.dump(res_json, f)


def run_eval(datasets, run_id):
    """
    Evaluate the performance for given datasets for checkpoints with run_id.

    Args:
        datasets: list of dataset names
        run_id: run_id of checkpoints to be used
    """
    config = {
        "weights_path": Path(r"./weights"),
        "datasets_folder": Path("./datasets"),
        "results_save_path": Path("./eval_res"),
        "image_save_path": None,  # set to save images
        "score_save_path": None,  # set to save scores
        "seed": 42,
        "batch": 8,
        "num_workers": 8,
        "run_id": str(run_id),
    }
    data_functions = {
        "sensum": get_sensum,
        "ksdd2": get_ksdd2,
        "mvtec": get_mvtec,
        "visa": get_visa,
    }

    for dataset in datasets:
        data_list = data_functions[dataset](config)

        results_writer = ResultsWriter(
            metrics=[
                "AP-det",
                "AP-loc",
                "P-AUROC",
                "I-AUROC",
                "AUPRO",
                "seg-AP-det",
                "seg-I-AUROC",
                "run_id",
            ]
        )

        for cat, datamodule in data_list:
            print("Evaluating", f"{dataset}-{cat}")
            weight_path = (
                config["weights_path"] / config["run_id"] / dataset / cat / "weights.pt"
            )
            model = SuperSimpleNet(image_size=datamodule.image_size, config=config)
            model.load_model(weight_path)

            image_metrics = {
                "I-AUROC": AUROC(),
                "AP-det": AveragePrecision(num_classes=1),
            }
            pixel_metrics = {
                "P-AUROC": AUROC(),
                "AP-loc": AveragePrecision(num_classes=1),
                "AUPRO": AUPRO(),  # aupro calculation can be slow, and it requires some gpu memory
            }

            results = eval(
                model=model,
                datamodule=datamodule,
                device="cuda",
                image_metrics=image_metrics,
                pixel_metrics=pixel_metrics,
                normalize=True,
                score_save_path=config["score_save_path"]
                / config["run_id"]
                / dataset
                / cat
                if config["score_save_path"]
                else None,
                image_save_path=config["image_save_path"]
                / config["run_id"]
                / dataset
                / cat
                if config["image_save_path"]
                else None,
            )
            results["run_id"] = config["run_id"]

            if dataset == "sensum":
                # for sensum remove fold num when saving
                res_cat = cat[:-2]
            else:
                res_cat = cat
            results_writer.add_result(
                category=res_cat,
                last=results,
            )
            results_writer.save(
                Path(config["results_save_path"]) / config["run_id"] / dataset
            )


if __name__ == "__main__":
    run_eval(datasets=["mvtec", "visa", "ksdd2", "sensum"], run_id=0)
    # to get mean and std of multiple runs, specify them with run_ids
    generate_result_json(
        run_ids=["0"],
        datasets=["mvtec", "visa", "ksdd2", "sensum"],
        res_path=Path("./eval_res"),
    )
