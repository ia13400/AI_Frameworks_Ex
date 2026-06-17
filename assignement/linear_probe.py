import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import FLAT_PAIRS_PATH, PROBE_SPLIT_PATH
from model_utils import section
from plotting_utils import plot_logistic_regression_probabilities


PROBE_SPLIT_STRATEGY = "stratified_all_hu_liu_one_token_words_v1"


def split_word_record(item):
    return {
        "word": item["word"],
        "token": item["token"],
        "token_id": int(item["token_id"]),
        "sentiment": item["sentiment"],
    }


def load_or_create_probe_split(positive_words, negative_words):
    all_probe_words = sorted(
        positive_words + negative_words,
        key=lambda item: (item["sentiment"], item["word"], item["token_id"]),
    )

    use_saved_split = False
    if PROBE_SPLIT_PATH.exists():
        split_payload = json.loads(PROBE_SPLIT_PATH.read_text(encoding="utf-8"))
        use_saved_split = split_payload.get("split_strategy") == PROBE_SPLIT_STRATEGY

    if use_saved_split:
        train_words = split_payload["train"]
        test_words = split_payload["test"]
        print(f"\nLoaded fixed logistic-regression split from {PROBE_SPLIT_PATH}")
    else:
        if PROBE_SPLIT_PATH.exists():
            print(
                "\nExisting logistic-regression split uses an older strategy; "
                "regenerating it with all one-token Hu & Liu words."
            )

        y_all = np.array([1 if item["sentiment"] == "positive" else 0 for item in all_probe_words])
        split_train_indices, split_test_indices = train_test_split(
            np.arange(len(all_probe_words)),
            test_size=0.2,
            stratify=y_all,
            random_state=42,
        )
        train_words = [split_word_record(all_probe_words[index]) for index in split_train_indices]
        test_words = [split_word_record(all_probe_words[index]) for index in split_test_indices]
        split_payload = {
            "source": "Hu & Liu one-token sentiment words",
            "split_strategy": PROBE_SPLIT_STRATEGY,
            "random_state": 42,
            "test_size": 0.2,
            "stratify": "sentiment label",
            "label_mapping": {"negative": 0, "positive": 1},
            "all_one_token_word_counts": {
                "positive": int((y_all == 1).sum()),
                "negative": int((y_all == 0).sum()),
                "total": int(len(y_all)),
            },
            "train": train_words,
            "test": test_words,
        }
        PROBE_SPLIT_PATH.write_text(
            json.dumps(split_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved fixed logistic-regression split to {PROBE_SPLIT_PATH}")

    probe_words = train_words + test_words
    train_indices = np.arange(len(train_words))
    test_indices = np.arange(len(train_words), len(probe_words))
    return probe_words, train_indices, test_indices


def print_probe_evaluation(split_name, y_true, y_pred):
    print(f"\n--- Logistic Regression Linear Probe Evaluation: {split_name} ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 score : {f1_score(y_true, y_pred):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))


def print_misclassified_test_words(test_word_records, y_test, y_test_pred, y_test_prob):
    misclassified = [
        {
            "word": item["word"],
            "true": "positive" if true_label == 1 else "negative",
            "predicted": "positive" if predicted_label == 1 else "negative",
            "probability_positive": float(probability),
        }
        for item, true_label, predicted_label, probability in zip(
            test_word_records,
            y_test,
            y_test_pred,
            y_test_prob,
        )
        if true_label != predicted_label
    ]

    print("\nMisclassified test words:")
    print("-" * 72)
    if misclassified:
        for row in sorted(misclassified, key=lambda item: item["word"]):
            print(
                f"{row['word']:<18} "
                f"true={row['true']:<8} "
                f"predicted={row['predicted']:<8} "
                f"p_positive={row['probability_positive']:.4f}"
            )
    else:
        print("None")


def plot_field_word_probabilities(classifier, scaler, embedding_matrix):
    if not FLAT_PAIRS_PATH.exists():
        print(f"Skipping logistic regression probability plot: {FLAT_PAIRS_PATH} not found.")
        return

    field_pairs = json.loads(FLAT_PAIRS_PATH.read_text(encoding="utf-8"))["pairs"]
    field_words_by_key = {}
    for pair in field_pairs:
        for sentiment in ["positive", "negative"]:
            item = pair[sentiment]
            normalized_word = "".join(item["word"].split()).lower()
            field_words_by_key.setdefault(normalized_word, item)

    field_words = list(field_words_by_key.values())
    field_word_ids = torch.tensor([item["token_id"] for item in field_words], dtype=torch.long)
    field_word_vectors = embedding_matrix[field_word_ids].float().numpy()
    field_word_vectors_scaled = scaler.transform(field_word_vectors)
    probability_positive = classifier.predict_proba(field_word_vectors_scaled)[:, 1]

    probability_rows = sorted(
        [
            {
                "word": item["word"],
                "sentiment": item["sentiment"],
                "probability_positive": float(probability),
            }
            for item, probability in zip(field_words, probability_positive)
        ],
        key=lambda row: row["probability_positive"],
    )
    plot_logistic_regression_probabilities(probability_rows)


def run_logistic_regression_probe(sentiment_state):
    section(8, "Logistic Regression Sentiment Linear Probe")

    # Logistic regression is used as a linear probe. It tests whether a simple
    # linear decision boundary can separate sentiment labels in embedding space.
    embedding_matrix = sentiment_state["embedding_matrix"]
    positive_words = sentiment_state["positive_words"]
    negative_words = sentiment_state["negative_words"]

    probe_words, train_indices, test_indices = load_or_create_probe_split(
        positive_words,
        negative_words,
    )

    X = embedding_matrix[
        torch.tensor([item["token_id"] for item in probe_words], dtype=torch.long)
    ].float().numpy()
    y = np.array([1 if item["sentiment"] == "positive" else 0 for item in probe_words])

    print(f"\nX.shape = {X.shape}")
    print(f"y.shape = {y.shape}")
    print(f"Positive labels: {int(y.sum())}")
    print(f"Negative labels: {int((y == 0).sum())}")
    print(f"Train examples: {len(train_indices)}")
    print(f"Test examples : {len(test_indices)}")

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    print(
        "Train labels  : "
        f"positive={int(y_train.sum())}, negative={int((y_train == 0).sum())}"
    )
    print(
        "Test labels   : "
        f"positive={int(y_test.sum())}, negative={int((y_test == 0).sum())}"
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = LogisticRegression(max_iter=2000, random_state=42)
    classifier.fit(X_train_scaled, y_train)

    y_test_pred = classifier.predict(X_test_scaled)
    y_test_prob = classifier.predict_proba(X_test_scaled)[:, 1]
    y_train_pred = classifier.predict(X_train_scaled)

    print_probe_evaluation("test set", y_test, y_test_pred)
    test_word_records = [probe_words[index] for index in test_indices]
    print_misclassified_test_words(test_word_records, y_test, y_test_pred, y_test_prob)
    print_probe_evaluation("train set", y_train, y_train_pred)

    print(
        "\nInterpretation: high accuracy indicates that sentiment information is "
        "linearly separable in the input embedding space."
    )

    plot_field_word_probabilities(classifier, scaler, embedding_matrix)

    return classifier, scaler
