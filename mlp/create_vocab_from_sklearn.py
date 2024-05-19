import pickle
import json


if __name__ == "__main__":
    with open("../classic/experiments/logreg_tfidf/runs/2024-05-19-06-20-20/models/pipeline.pkl", "rb") as file:
        vectorizer = pickle.load(file)[0]
    with open("vocab_1_3.json", "w") as file:
        vocab = {k: int(v) for k, v in vectorizer.vocabulary_.items()}
        json.dump(vocab, file, indent=4)
