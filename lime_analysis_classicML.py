#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lime.lime_text import LimeTextExplainer

LABEL_TO_ID = {
    "시설 결함": 0,
    "안전기준 미준수": 1,
    "운송차량": 2,
}
CLASS_NAMES = ["C1_시설_결함", "C2_안전기준_미준수", "C3_운송차량"]


def load_stopwords(data_directory):
    path = os.path.join(data_directory, "korean_stopwords.txt")
    if not os.path.exists(path):
        print(f"[WARN] Stopword file not found: {path}")
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def build_preprocessor(data_directory, disable_okt=False):
    stopwords = load_stopwords(data_directory)

    if disable_okt:
        okt = None
        print("[WARN] --disable_okt enabled. Preprocessing may differ from training.")
    else:
        from konlpy.tag import Okt
        okt = Okt()

    def preprocess(text):
        text = "" if pd.isna(text) else str(text)
        text = re.sub(r"\b\d+\b", "", text)
        tokens = okt.morphs(text, stem=True) if okt is not None else text.split()
        tokens = [t for t in tokens if t not in stopwords]
        return " ".join(tokens)

    return preprocess


def load_test_dataframe(data_directory, test_file):
    path = os.path.join(data_directory, test_file)
    df = pd.read_csv(path, encoding="utf-8")
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in test csv: {missing}. Current columns: {list(df.columns)}")

    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(str).str.strip()

    unknown = sorted(set(df["label"]) - set(LABEL_TO_ID.keys()))
    if unknown:
        raise ValueError(f"Unknown labels in test set: {unknown}. Allowed labels: {list(LABEL_TO_ID.keys())}")

    return df


def load_checkpoint(model_directory, model_name):
    path = os.path.join(model_directory, "checkpoint", f"{model_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    saved = joblib.load(path)
    return saved["model"], saved["vectorizer"], path


def make_predict_proba_fn(model, vectorizer, preprocess):
    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba(). For SVM, use SVC(probability=True).")

    def predict_proba(texts):
        processed = [preprocess(t) for t in texts]
        X = vectorizer.transform(processed)
        return model.predict_proba(X)

    return predict_proba


def select_indices(df, selection, num_explain, seed):
    if selection == "all":
        return df.index.tolist()
    if selection == "first":
        return df.index[:num_explain].tolist()
    if selection == "random":
        return df.sample(n=min(num_explain, len(df)), random_state=seed).index.tolist()
    if selection == "misclassified":
        candidates = df[df["y_true"] != df["y_pred"]]
        return candidates.head(num_explain).index.tolist()
    if selection == "correct":
        candidates = df[df["y_true"] == df["y_pred"]]
        return candidates.head(num_explain).index.tolist()
    if selection == "high_confidence":
        return df.sort_values("pred_confidence", ascending=False).head(num_explain).index.tolist()
    if selection == "low_confidence":
        return df.sort_values("pred_confidence", ascending=True).head(num_explain).index.tolist()
    raise ValueError(f"Unknown selection: {selection}")


def main():
    parser = argparse.ArgumentParser(description="LIME analysis for classic ML text classifiers")
    parser.add_argument("--data_directory", required=True)
    parser.add_argument("--test_file", default="test/test.csv")
    parser.add_argument("--model_directory", required=True)
    parser.add_argument("--model", required=True, choices=["LR", "RF", "SVM", "XGBoost"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_explain", default=20, type=int)
    parser.add_argument(
        "--selection",
        default="misclassified",
        choices=["all", "first", "random", "misclassified", "correct", "high_confidence", "low_confidence"],
    )
    parser.add_argument("--lime_num_features", default=12, type=int)
    parser.add_argument("--lime_num_samples", default=1000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--disable_okt", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)

    out_dir = Path(args.output_dir or os.path.join(args.model_directory, "lime_result"))
    html_dir = out_dir / "html"
    out_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    preprocess = build_preprocessor(args.data_directory, disable_okt=args.disable_okt)
    test_df = load_test_dataframe(args.data_directory, args.test_file)
    model, vectorizer, ckpt_path = load_checkpoint(args.model_directory, args.model)
    predict_proba = make_predict_proba_fn(model, vectorizer, preprocess)

    test_df["processed_text"] = test_df["text"].apply(preprocess)
    test_df["y_true"] = test_df["label"].map(LABEL_TO_ID).astype(int)

    probs = predict_proba(test_df["text"].tolist())
    test_df["y_pred"] = probs.argmax(axis=1)
    test_df["pred_confidence"] = probs.max(axis=1)
    test_df["true_label"] = [CLASS_NAMES[i] for i in test_df["y_true"]]
    test_df["pred_label"] = [CLASS_NAMES[i] for i in test_df["y_pred"]]

    for class_idx, class_name in enumerate(CLASS_NAMES):
        test_df[f"prob_{class_name}"] = probs[:, class_idx]

    test_df.to_csv(out_dir / "test_predictions_for_lime.csv", index=False, encoding="utf-8-sig")

    acc = accuracy_score(test_df["y_true"], test_df["y_pred"])
    report = classification_report(test_df["y_true"], test_df["y_pred"], target_names=CLASS_NAMES, digits=4, zero_division=0)
    cm = confusion_matrix(test_df["y_true"], test_df["y_pred"])

    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint_path": ckpt_path,
            "accuracy": float(acc),
            "class_names": CLASS_NAMES,
            "confusion_matrix": cm.tolist(),
            "selection": args.selection,
            "num_explain": args.num_explain,
            "lime_num_features": args.lime_num_features,
            "lime_num_samples": args.lime_num_samples,
        }, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Accuracy: {acc:.6f}")
    print(report)

    explainer = LimeTextExplainer(
        class_names=CLASS_NAMES,
        split_expression=r"\s+",
        bow=True,
        random_state=args.seed,
    )

    selected_indices = select_indices(test_df, args.selection, args.num_explain, args.seed)
    rows = []

    for rank, idx in enumerate(selected_indices, start=1):
        row = test_df.loc[idx]
        pred_label_idx = int(row["y_pred"])

        exp = explainer.explain_instance(
            text_instance=row["text"],
            classifier_fn=predict_proba,
            labels=[pred_label_idx],
            num_features=args.lime_num_features,
            num_samples=args.lime_num_samples,
        )

        html_path = html_dir / f"lime_idx_{idx}_true_{row['true_label']}_pred_{row['pred_label']}.html"
        exp.save_to_file(str(html_path))

        for token, weight in exp.as_list(label=pred_label_idx):
            rows.append({
                "test_index": int(idx),
                "rank": int(rank),
                "token": token,
                "lime_weight": float(weight),
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
                "pred_confidence": float(row["pred_confidence"]),
                "is_correct": bool(row["y_true"] == row["y_pred"]),
                "html_path": str(html_path),
                "raw_text": row["text"],
                "processed_text": row["processed_text"],
            })

        print(f"[{rank}/{len(selected_indices)}] idx={idx}, true={row['true_label']}, pred={row['pred_label']}, html={html_path}")

    pd.DataFrame(rows).to_csv(out_dir / "lime_token_weights.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved LIME results to: {out_dir}")


if __name__ == "__main__":
    main()
