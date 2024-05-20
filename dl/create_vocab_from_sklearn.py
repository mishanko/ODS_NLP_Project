import pickle
import json
from fire import Fire


def main(path: str, save_name: str):
    with open(path, "rb") as file:
        vectorizer = pickle.load(file)[0]
    print(f"[INFO] Vocab size: {len(vectorizer.vocabulary_)}")
    with open(f"data/{save_name}.json", "w") as file:
        vocab = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
        json.dump(vocab, file, indent=4)


if __name__ == "__main__":
    Fire(main)
    # with open("../classic/experiments/logreg_tfidf/runs/2024-05-19-06-20-20/models/pipeline.pkl", "rb") as file:
    #     vectorizer = pickle.load(file)[0]
    # print(f"[INFO] Vocab size: {len(vectorizer.vocabulary_)}")
    # with open("data/vocab_1_3.json", "w") as file:
    #     vocab = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
    #     json.dump(vocab, file, indent=4)
