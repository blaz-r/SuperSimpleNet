import sys
from pathlib import Path
from anomalib.data.utils import download_and_extract, DownloadInfo


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "../../datasets/KolektorSDD2"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    DOWNLOAD_INFO = DownloadInfo(
        name="KolektorSDD2",
        filename="KolektorSDD2.zip",
        url="https://go.vicos.si/kolektorsdd2",
        hash="1ed0f628122776ace4965afa30dfe316",
    )

    if path.exists():
        raise ValueError(f"KolektorSDD2 already exists at {path}")

    download_and_extract(path, DOWNLOAD_INFO)


if __name__ == "__main__":
    main()
