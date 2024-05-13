from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import shutil

N = int(1e6)
RND = 42


if __name__ == "__main__":
    projdir = Path("/data_research/Projects/nlp-proj/guesslang_data")
    datadir = projdir / "files"
    df = pd.read_csv(projdir / "11_extracted_files.csv")
    fnames = [x.name for x in datadir.glob("*/*")]
    df = df[df["extract_to"].isin(fnames)]
    df = df.sample(n=N, random_state=RND)

    savedir = projdir / "Dataset"
    savedir_data = savedir / "Data"
    savedir_data.mkdir(exist_ok=True, parents=True)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        fname = row["extract_to"]
        split = row["usage"]
        cursd = savedir_data / split
        if not cursd.exists():
            cursd.mkdir()
        shutil.move(datadir / split / fname, cursd / fname)
    df.to_csv(savedir / "Annotation.csv", index=False)
