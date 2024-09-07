import csv
import gc
import sys
import timeit

from tqdm import tqdm

sys.path.append("../")

from model.supersimplenet import SuperSimpleNet
import torch


def params():
    config = {
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "noise_std": 0.015,
        "stop_grad": False,
    }
    model = SuperSimpleNet(image_size=(256, 256), config=config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total params:", total_params, "Trainable params:", trainable_params)


def prepare_image(batched=False):
    if batched:
        img = torch.randn(16, 3, 256, 256, dtype=torch.float16)
    else:
        img = torch.randn(1, 3, 256, 256, dtype=torch.float16)

    return img


def prepare_model():
    config = {
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "noise_std": 0.015,
        "stop_grad": False,
    }
    model = SuperSimpleNet(image_size=(256, 256), config=config)
    # model.load_model("./pcb1/weights.pt")
    model.to("cuda")
    model.to(torch.float16)
    model.eval()

    return model


@torch.no_grad()
def inference_speed(reps=1000):
    model = prepare_model()
    img = prepare_image()

    # first - warmup
    for i in tqdm(range(reps), desc="Warmup"):
        img = img.to("cpu")

        img = img.to("cuda")
        out = model(img)
        out = out[0].to("cpu"), out[1].to("cpu")

    total_time = 0
    # next - real
    for i in tqdm(range(reps), desc="Timing inference"):
        img = img.to("cpu")

        t0 = timeit.default_timer()

        img = img.to("cuda")
        out = model(img)
        out = out[0].to("cpu"), out[1].to("cpu")

        t1 = timeit.default_timer()
        total_time += t1 - t0

    # * 1000 to get ms
    ms = total_time * 1000 / reps
    print("Speed in ms:", ms)
    return ms


@torch.no_grad()
def throughput(reps=1000):
    model = prepare_model()
    img = prepare_image(batched=True)

    # first - warmup
    for i in tqdm(range(reps), desc="Warmup"):
        img = img.to("cpu")

        img = img.to("cuda")
        out = model(img)
        out = out[0].to("cpu"), out[1].to("cpu")

    total_time = 0
    # next - real
    for i in tqdm(range(reps), desc="Throughput"):
        img = img.to("cpu")

        t0 = timeit.default_timer()

        img = img.to("cuda")
        out = model(img)
        out = out[0].to("cpu"), out[1].to("cpu")

        t1 = timeit.default_timer()
        total_time += t1 - t0

    thru = 16 * reps / total_time
    print("Throughput:", thru)
    return thru


@torch.no_grad()
def memory(reps=1000):
    model = prepare_model()
    img = prepare_image()
    img = img.to("cuda")

    # first - warmup
    for i in tqdm(range(reps), desc="Warmup"):
        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        out = model(img)

        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_memory = 0
    # next - real
    for i in tqdm(range(reps), desc="Memory calc"):
        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        out = model(img)

        total_memory += torch.cuda.max_memory_reserved()

        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # MB -> 10**6 bytes, then "reps" runs
    mbs = total_memory / (10**6) / reps
    print("Memory in MB:", mbs)
    return mbs


@torch.no_grad()
def flops(reps=1000):
    model = prepare_model()
    img = prepare_image()
    img = img.to("cuda")

    # first - warmup
    out = model(img)

    # real - don't need reps as the result is always same
    with torch.profiler.profile(with_flops=True) as prof:
        out = model(img)
    tflops = sum(x.flops for x in prof.key_averages()) / 1e9
    print("TFLOPS:", tflops)

    return tflops


def main():
    cycles = 6
    reps = 1000

    torch.backends.cudnn.deterministic = True

    with open(f"perf_{sys.argv[1]}.csv", "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["time", "throughput", "memory", "tflops"])
        for cyc in range(cycles):
            ms = inference_speed(reps)
            thru = throughput(reps)
            mbs = memory(reps)
            tflops = flops(reps)

            if cyc == 0:
                # skip first one, as the system is not warmed up and it's too fast
                continue

            writer.writerow([ms, thru, mbs, tflops])
            print("-" * 42)
            print("Speed [ms]:", ms)
            print("Throughput:", thru)
            print("Memory [MB]:", mbs)
            print("TFLOPS:", tflops)
            print("-" * 42)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python perf_main.py <gpu_model>")

    main()
    # params()
