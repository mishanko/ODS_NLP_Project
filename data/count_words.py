import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

N = int(2e5)
RND = 42
TOPK = 50


class RegexPreprocessor:
    pattern = r'([a-zA-Z_][a-zA-Z0-9_]*|\b\d+\b|".*?"|[\+\-\*/=<>{}\[\](),.;:]+|\s+)'

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


def read_dataset(files: iter):
    return [read_file(path) for path in tqdm(files)]


def get_common_words(files: list[str], topk: int = 50) -> dict[str, dict[str, int]]:
    counter = defaultdict(Counter)
    preprocessor = RegexPreprocessor()
    for file in tqdm(files):
        ext = file.name.split(".")[-1].lower()
        string = read_file(file)
        strings = preprocessor(string)
        cur_counter = Counter(strings)
        counter[ext] += cur_counter
    counter = {k: dict(v.most_common(topk)) for k, v in counter.items()}
    return counter


if __name__ == "__main__":
    projdir = Path("/data_research/Projects/nlp-proj/guesslang_data/Dataset")
    annotation = pd.read_csv(projdir / "Annotation.csv")
    annotation = annotation[annotation["usage"] == "train"].sample(
        n=N, random_state=RND
    )
    datadir = projdir / "Data/train"
    files = [datadir / row["extract_to"] for i, row in annotation.iterrows()]
    counter = get_common_words(files, topk=TOPK)
    with open("common_words.json", "w") as f:
        json.dump(counter, f, indent=4)
