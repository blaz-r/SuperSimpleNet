import sys
from pathlib import Path
from anomalib.data.visa import Visa


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "../../datasets/visa"
    path = Path(path)

    # use anomalib to download, extract and make split inside "visa_pytorch"
    visa = Visa(root=path, category="candle", image_size=(256, 256))
    visa.prepare_data()


if __name__ == "__main__":
    main()
