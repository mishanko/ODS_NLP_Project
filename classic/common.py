from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from tqdm.contrib.concurrent import process_map


def read_file(path: Path) -> tuple[str, str]:
    with open(path, "r") as f:
        text = f.read()
    return text, path.suffix


def read_dataset(files: list[Path], max_workers: int = 1) -> tuple[list[str], list[str]]:
    dataset = process_map(read_file, files, desc="Reading dataset ...", max_workers=max_workers, chunksize=2**8)
    X = [x[0] for x in dataset]
    y = [x[1] for x in dataset]
    return X, y


def create_metrics_df(y_test: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> pd.DataFrame:
    pr, re, f1, support = precision_recall_fscore_support(y_test, y_pred)
    df = []
    for i, name in enumerate(classes):
        ap = average_precision_score(y_test == i, y_pred == i)
        df.append(
            {
                "label": name,
                "precision": pr[i],
                "recall": re[i],
                "f1_score": f1[i],
                "support": support[i],
                "average_precision": ap,
            }
        )
    pr, re, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    df.append({"label": "macro_average", "precision": pr, "recall": re, "f1_score": f1})
    df = pd.DataFrame(df)
    return df
