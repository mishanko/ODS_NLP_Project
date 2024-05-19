from pathlib import Path
from common import read_dataset, create_metrics_df
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import hydra
import pickle
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd


def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("as_tuple", resolve_tuple)


def fit_once(transformer):
    fitted = [False]

    def func(x):
        if not fitted[0]:
            transformer.fit(x)
            fitted[0] = True
        return transformer.transform(x)

    return FunctionTransformer(func)


@hydra.main(version_base=None, config_path="config", config_name="train")
def train(config: DictConfig) -> None:
    if not config.exp_name:
        raise ValueError("`exp_name` must be set")
    datadirs = {k: Path(v) for k, v in config.datadirs.items()}
    X_train, y_train = read_dataset(list(datadirs["train"].glob("*.*")), config.max_workers)
    le = LabelEncoder()
    # if config.vectorizer_path:
    #     with open(config.vectorizer_path, "rb") as file:
    #         vectorizer = pickle.load(file)[0]

    # else:
    vectorizer = instantiate(config.vectorizer)
    pipeline = Pipeline(
        [
            ("vectorizer", vectorizer),
            ("model", instantiate(config.model)),
        ]
    )

    y_train = le.fit_transform(y_train)
    pipeline.fit(X_train, y_train)
    outputdir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    modeldir = outputdir / "models"
    modeldir.mkdir()

    with open(modeldir / "pipeline.pkl", "wb") as file:
        pickle.dump(pipeline, file)
    with open(modeldir / "label_encoder.pkl", "wb") as file:
        pickle.dump(le, file)

    X_test, y_test = read_dataset(list(datadirs["test"].glob("*.*")), config.max_workers)
    y_test = le.transform(y_test)
    preds = pipeline.predict(X_test)
    df = create_metrics_df(y_test, preds, le.classes_)
    df.to_csv(outputdir / "test_metrics.csv", index=False)


if __name__ == "__main__":
    train()
