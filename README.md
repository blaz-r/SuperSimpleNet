# [ICPR 2024] SuperSimpleNet

Official implementation of [SuperSimpleNet : Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection](https://arxiv.org/abs/2408.03143) - ICPR 2024.

## Environment
```bash
conda create -n ssn_env python=3.10
pip install -r requirements.txt
```

The project uses wandb for logging, but it's optional. 
To enable this: uncomment wandb from requirements.txt to install and set `LOG_WANDB=True` at the top of train.py.

## Datasets

Follow the steps below to prepare all 4 datasets used in the paper. The code used to download datasets requires the env from the previous step.
If you already have the files prepared for a specific dataset, you can change the path in `eval.py`/`train.py`.

Note that for the VisA, the data needs to be correctly split and stored inside `visa/visa_pytorch`. 
This is handled automatically with the provided script. Ensure that the splits are correct if you are using existing VisA data.

1. Change directory to `./datamodules/setup/`.
2. Run `prepare_mvtec.py` to download and extract MVTec files.
3. Run `prepare_visa.py` to download, extract, and **prepare splits** for VisA files.
4. Run `prepare_ksdd2.py` to download and extract KSDD2 files.
5. To download SensumSODF, request a link on the official site.
   - Download the data from the link you receive [here](https://www.sensum.eu/sensumsodf-dataset/) and extract it to the dataset folder.
   
   - Then download SensumSODF 3-fold CV [split files](https://drive.google.com/file/d/1CrolrOHHm3wHaKu6JKqQ62qQGclwDKBM/view?usp=sharing). Extract them and place the `sensum_splits` folder inside the SensumSODF root.
   
   - If you are evaluating your method on SensumSODF, use the provided split files within the 3-fold CV setting for fair comparison.

The final structure should then look like this (case-sensitive):

```
datasets/
    KolektorSDD2/
        train/...
        test/...
        split_weakly_0.pyb
        ...
    SensumSODF/
        capsule/...
        softgel/...
        sensum_splits/
            capsule/
                0/...
                ...
            softgel/...
    mvtec/
         bottle/...
         ...
    visa/
        visa_pytorch/
            candle/
            ....
```


## Checkpoints

Checkpoints are available [here](https://drive.google.com/file/d/1pCfBxCGXdsN0LVuf4R0KIVE6oRXJwMJ5/view?usp=sharing). 
Extract them into `./weights` path and ensure they are all inside a directory with run_id 0: 
```
./weights/
   0/
      ksdd2/
      sensum/
      mvtec/
      visa/
```

We report an average of 5 runs in our paper, but the weights from the link are only for the best run.
Therefore, the results won't exactly match the ones reported in the paper.

We also include the reported mean and std as a json inside `paper_results` for all datasets in the paper.

## Evaluate

Evaluate using the checkpoints:

```bash
python eval.py
```

Slurm script `run_slurm_eval.sh` is also provided to execute evaluation on a slurm based system.

---
Config for the model and datasets is contained within the eval.py file. 

## Train

Train the model:

```bash
python train.py <dataset_name>
```
Possible dataset names are: `mvtec`, `visa`, `sensum`, and `ksdd2`.

Slurm script `run_slurm_train.sh` is also provided to execute training on a slurm based system.

---

Config for the model and datasets is contained within train.py file. If you want to modify training params, change the values there. 

We recommend taking the MVTec parameters when training on your own **unsupervised** dataset and SenumSODF parameters for **supervised** dataset.

## Performance benchmark

Use the code inside `./perf` to evaluate performance metrics (inference speed, throughput, memory consumption, flops):

```bash
python perf_main.py <gpu_model>.
```

Slurm script `run_slurm_perf.sh` is also provided to execute benchmark on slurm based system.

Note that the results in paper are obtained with AMD Epyc 7272 CPU and NVIDIA Tesla V100S GPU and might therefore differ from the ones obtained on your system.

We also include the performance results from the paper inside `paper_results`.

## Citation

```bibtex
@InProceedings{rolih2024supersimplenet,
  author={Rolih, Bla{\v{z}} and Fu{\v{c}}ka, Matic and Sko{\v{c}}aj, Danijel},
  booktitle={International Conference on Pattern Recognition}, 
  title={{S}uper{S}imple{N}et: {U}nifying {U}nsupervised and {S}upervised {L}earning for {F}ast and {R}eliable {S}urface {D}efect {D}etection},
  year={2024}
}
```

## Acknowledgement

Thanks to [SimpleNet](https://github.com/DonaldRR/SimpleNet) for great inspiration.
