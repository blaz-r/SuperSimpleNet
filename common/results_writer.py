from pathlib import Path

import pandas as pd


class ResultsWriter:
    def __init__(self, metrics: list[str] = None):
        self.columns = ["category"] + metrics
        self.last_results = []

    def add_result(self, category: str, last: dict):
        last["category"] = category
        self.last_results.append(last)

    def save(self, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        last_df = pd.DataFrame(self.last_results, columns=self.columns)
        last_df.to_csv(path / "last.csv", index=False)

        avg = last_df.mean(axis=0, numeric_only=True)
        avg["category"] = "avg"
        # transpose to get names in columns for csv
        avg = avg.to_frame().T
        avg.to_csv(path / "avg.csv", index=False)
