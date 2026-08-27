"""Local paper-style baselines for phishing/spam email classification.

This script does not call any AI API. It evaluates TF-IDF and Word2Vec feature
pipelines on the same cleaned dataset used by the DeepSeek experiment.

Labels: 0 = legitimate/ham, 1 = spam or phishing.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from dataset_loader import load_sources, sample_data


TOKEN_RE = re.compile(r"[A-Za-z0-9_@$.'/-]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def decision_scores(model: object, features: object) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    scores = model.decision_function(features)
    return 1.0 / (1.0 + np.exp(-scores))


def save_results(
    *,
    output: Path,
    experiment: str,
    model: object,
    test_frame,
    prediction: np.ndarray,
    score: np.ndarray,
    train_size: int,
) -> None:
    experiment_dir = output / experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        test_frame["label"], prediction, output_dict=True, digits=4
    )
    matrix = confusion_matrix(test_frame["label"], prediction).tolist()
    metrics = {
        "experiment": experiment,
        "train_size": train_size,
        "test_size": len(test_frame),
        "confusion_matrix": matrix,
        "classification_report": report,
    }
    (experiment_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    predictions = test_frame[["source", "label", "text_hash"]].copy()
    predictions["prediction"] = prediction
    predictions["spam_score"] = score
    predictions["correct"] = predictions["label"] == predictions["prediction"]
    predictions.to_csv(experiment_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump(model, experiment_dir / "model.joblib")
    print(f"\n[{experiment}]")
    print(classification_report(test_frame["label"], prediction, digits=4))
    print("Confusion matrix [[TN, FP], [FN, TP]]:", matrix)


def run_tfidf(
    *,
    train,
    test,
    output: Path,
    max_features: int,
    include_rf: bool,
    seed: int,
) -> None:
    classifiers = {
        "tfidf_linear_svm": LinearSVC(class_weight="balanced", random_state=seed),
        "tfidf_multinomial_nb": MultinomialNB(),
    }
    if include_rf:
        classifiers["tfidf_random_forest"] = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )

    for name, classifier in classifiers.items():
        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=(1, 2),
                        max_features=max_features,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(train["text_combined"], train["label"])
        prediction = pipeline.predict(test["text_combined"])
        score = decision_scores(pipeline, test["text_combined"])
        save_results(
            output=output,
            experiment=name,
            model=pipeline,
            test_frame=test,
            prediction=prediction,
            score=score,
            train_size=len(train),
        )


def sentence_vectors(sentences: list[list[str]], model: object, vector_size: int) -> np.ndarray:
    vectors = np.zeros((len(sentences), vector_size), dtype=np.float32)
    for row, tokens in enumerate(sentences):
        known = [model.wv[token] for token in tokens if token in model.wv]
        if known:
            vectors[row] = np.mean(known, axis=0)
    return vectors


def run_word2vec(
    *,
    train,
    test,
    output: Path,
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    include_rf: bool,
    seed: int,
) -> None:
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise SystemExit(
            "Word2Vec baseline requires gensim. Install it with: "
            "python -m pip install gensim"
        ) from exc

    train_tokens = [tokenize(text) for text in train["text_combined"]]
    test_tokens = [tokenize(text) for text in test["text_combined"]]
    w2v = Word2Vec(
        sentences=train_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,
        epochs=epochs,
        seed=seed,
    )
    x_train = sentence_vectors(train_tokens, w2v, vector_size)
    x_test = sentence_vectors(test_tokens, w2v, vector_size)

    classifiers = {
        "word2vec_linear_svm": LinearSVC(class_weight="balanced", random_state=seed),
    }
    if include_rf:
        classifiers["word2vec_random_forest"] = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )

    for name, classifier in classifiers.items():
        classifier.fit(x_train, train["label"])
        prediction = classifier.predict(x_test)
        score = decision_scores(classifier, x_test)
        save_results(
            output=output,
            experiment=name,
            model={
                "word2vec": w2v,
                "classifier": classifier,
                "tokenizer": "TOKEN_RE",
                "vector_size": vector_size,
            },
            test_frame=test,
            prediction=prediction,
            score=score,
            train_size=len(train),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("archive.zip"))
    parser.add_argument("--output", type=Path, default=Path("output/baselines"))
    parser.add_argument("--limit", type=int, help="Stratified sample size; omit for all emails")
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        choices=["all", "tfidf", "word2vec"],
        default="all",
        help="Feature pipeline to run",
    )
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--word2vec-size", type=int, default=100)
    parser.add_argument("--word2vec-window", type=int, default=5)
    parser.add_argument("--word2vec-min-count", type=int, default=2)
    parser.add_argument("--word2vec-epochs", type=int, default=5)
    parser.add_argument(
        "--include-rf",
        action="store_true",
        help="Also run Random Forest baselines. This can be slow on full data.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    data = sample_data(load_sources(args.archive, args.max_chars), args.limit, args.seed)
    summary = data.groupby(["source", "label"]).size().unstack(fill_value=0)
    print(summary.rename(columns={0: "ham", 1: "spam/phishing"}))
    print(f"Total after cleaning/deduplication: {len(data)}")
    data[["source", "label", "text_hash"]].to_csv(
        args.output / "dataset_manifest.csv", index=False
    )
    if args.prepare_only:
        return

    train, test = train_test_split(
        data, test_size=args.test_size, stratify=data["label"], random_state=args.seed
    )
    print(f"Train: {len(train)}  Test: {len(test)}")

    if args.methods in ("all", "tfidf"):
        run_tfidf(
            train=train,
            test=test,
            output=args.output,
            max_features=args.max_features,
            include_rf=args.include_rf,
            seed=args.seed,
        )
    if args.methods in ("all", "word2vec"):
        run_word2vec(
            train=train,
            test=test,
            output=args.output,
            vector_size=args.word2vec_size,
            window=args.word2vec_window,
            min_count=args.word2vec_min_count,
            epochs=args.word2vec_epochs,
            include_rf=args.include_rf,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
