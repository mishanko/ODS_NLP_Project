import pickle
from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
from common import create_metrics_df, read_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder


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
    X_train, y_train = read_dataset(
        list(datadirs["train"].glob("*.*")), config.max_workers
    )
    le = LabelEncoder()
    with open(
        "/home/dmitry.zarubin/ODS_NLP_Project/classic/experiments/logreg_tfidf/runs/2024-05-19-11-32-08/models/pipeline.pkl",
        "rb",
    ) as file:
        vectorizer = pickle.load(file)[0]
    pipeline = Pipeline([("model", instantiate(config.model))])

    y_train = le.fit_transform(y_train)
    X_train = vectorizer.transform(X_train)
    pipeline.fit(X_train, y_train)
    outputdir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    modeldir = outputdir / "models"
    modeldir.mkdir()
    pipeline = Pipeline(
        [
            ("vectorizer", vectorizer),
            ("model", pipeline[0]),
        ]
    )
    X_test, y_test = read_dataset(
        list(datadirs["test"].glob("*.*")), config.max_workers
    )
    y_test = le.transform(y_test)
    preds = pipeline.predict(X_test)
    df = create_metrics_df(y_test, preds, le.classes_)
    df.to_csv(outputdir / "test_metrics.csv", index=False)
    joblib.dump(pipeline, "pipeline.pkl.gz", compress=3)
    with open(modeldir / "label_encoder.pkl", "wb") as file:
        pickle.dump(le, file)


if __name__ == "__main__":
    train()
