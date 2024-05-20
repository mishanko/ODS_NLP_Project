import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm.contrib.concurrent import process_map

MAX_WORKERS = 10
CHUNKSIZE = 2**12
TOPK = 10000


class RegexPreprocessor:
    pattern = r"\w+|[!\"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]"

    def __init__(self, lower: bool = True):
        self.lower = lower

    def __call__(self, string: str) -> list[str]:
        if self.lower:
            string = string.lower()
        return re.findall(self.pattern, string, re.DOTALL)


def read_file(path: Path) -> str:
    with open(path, "r") as f:
        text = f.read()
    return text


def get_vocab_sample(file: Path):
    counts = Counter(RegexPreprocessor()(read_file(file)))
    total = counts.total()
    return Counter({k: v / total for k, v in counts.items()})


def get_vocab(files: list[Path]) -> dict[str, int]:
    vocabs = process_map(
        get_vocab_sample, files, max_workers=MAX_WORKERS, chunksize=CHUNKSIZE
    )
    total_counter = sum(vocabs, Counter())
    num_counters = len(vocabs)
    return {k: v / num_counters for k, v in total_counter.items()}


if __name__ == "__main__":
    projdir = Path("/data/guesslang_data/Dataset")
    annotation = pd.read_csv(projdir / "Annotation.csv")
    annotation = annotation[annotation["usage"] == "train"]
    datadir = projdir / "Data/train"
    files = [datadir / row["extract_to"] for i, row in annotation.iterrows()]
    vocab = get_vocab(files)
    with open("vocab.json", "w") as f:
        json.dump(vocab, f)
