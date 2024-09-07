import sys
from pathlib import Path
from anomalib.data.mvtec import MVTec


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "../../datasets/mvtec"
    path = Path(path)

    mvtec = MVTec(root=path, category="bottle", image_size=(256, 256))
    mvtec.prepare_data()


if __name__ == "__main__":
    main()
