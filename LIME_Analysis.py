# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer
from lime.lime_text import LimeTextExplainer

import models


LABEL_TO_ID = {
    '시설 결함': 0,
    '안전기준 미준수': 1,
    '운송차량': 2,
}

CLASS_NAMES = ['C1_시설_결함', 'C2_안전기준_미준수', 'C3_운송차량']

DISPLAY_MAPPING = {
    "시설 결함": "C1_시설_결함",
    "안전기준 미준수": "C2_안전기준_미준수",
    "운송차량": "C3_운송차량"
}


def load_stopwords(data_directory: str):
    stopword_path = os.path.join(data_directory, "korean_stopwords.txt")
    if not os.path.exists(stopword_path):
        print(f"[WARN] Stopword file not found: {stopword_path}")
        return set()

    with open(stopword_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def build_preprocessor(data_directory: str):
    korean_stopwords = load_stopwords(data_directory)

    def preprocess(text):
        text = "" if pd.isna(text) else str(text)
        text = re.sub(r"\b\d+\b", "", text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in korean_stopwords]
        return " ".join(tokens)

    return preprocess


def load_test_dataframe(data_directory: str, test_file: str):
    test_path = os.path.join(data_directory, test_file)
    test_df = pd.read_csv(test_path, encoding="utf-8")

    required_cols = {"text", "label"}
    missing_cols = required_cols - set(test_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in test csv: {missing_cols}. Current columns: {list(test_df.columns)}")

    test_df = test_df.dropna(subset=["text", "label"]).reset_index(drop=True)
    return test_df


def load_kobert_model(model_name: str, model_directory: str, device: str):
    model = models.prepare_model(model_name, device=device)

    ckpt_path = os.path.join(model_directory, model_name, "checkpoint", "ckpt.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Expected path format: <model_directory>/<model>/checkpoint/ckpt.pth"
        )

    checkpoint = torch.load(ckpt_path, map_location=device)

    # Existing training code saves {"net": model.state_dict(), "acc": ..., "epoch": ...}
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        model.load_state_dict(checkpoint["net"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model, ckpt_path


def make_kobert_predict_proba_fn(model, tokenizer, device: str, batch_size: int = 16, max_length: int = 128):

    def predict_proba(texts):
        all_probs = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

            all_probs.append(probs)

        return np.vstack(all_probs)

    return predict_proba

def make_ClassicalML_predict_proba_fn(model, vectorizer, preprocess):
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

def load_ClassicalML_checkpoint(model_directory, model_name):
    path = os.path.join(model_directory, "checkpoint", f"{model_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    saved = joblib.load(path)
    return saved["model"], saved["vectorizer"], path

def main():
    parser = argparse.ArgumentParser(description="LIME analysis for trained KoBERT classifier")
    parser.add_argument("--data_directory", required=True, type=str)
    parser.add_argument("--test_file", default="test/test.csv", type=str)
    parser.add_argument("--model_directory", required=True, type=str)
    parser.add_argument("--model", default="kobert", type=str)
    parser.add_argument("--output_dir", default=None, type=str)

    parser.add_argument("--num_explain", default=20, type=int)
    parser.add_argument(
        "--selection",
        default="misclassified",
        choices=["all", "first", "random", "misclassified", "correct", "high_confidence", "low_confidence"],
    )
    parser.add_argument("--lime_num_features", default=12, type=int)
    parser.add_argument("--lime_num_samples", default=1000, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--disable_okt", action="store_true")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join("lime_withoutOKT", args.model_directory, "lime_result")
        os.makedirs(output_dir, exist_ok=True)

    output_dir = Path(output_dir)
    html_dir = output_dir / "html"
    output_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    class_labels = list(LABEL_TO_ID.keys())
    class_names = CLASS_NAMES

    print("[INFO] LabelEncoder classes_ order:")
    for idx, name in enumerate(class_labels):
        print(f"Allowed labels: {class_labels}")
    preprocess = build_preprocessor(args.data_directory)

    test_df = load_test_dataframe(args.data_directory, args.test_file)
    test_df["label"] = test_df["label"].astype(str).str.strip()
    unknown_labels = sorted(set(test_df["label"]) - set(LABEL_TO_ID.keys()))
    if unknown_labels:
        raise ValueError(
            f"Test set contains labels not used by the model: {unknown_labels}\n"
            f"Allowed labels: {class_labels}"
        )
    test_df["processed_text"] = test_df["text"].apply(preprocess)
    test_df["y_true"] = test_df["label"].map(LABEL_TO_ID).astype(int)
    

    if args.model=="kobert":
        tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", use_fast=False)
        model, ckpt_path = load_kobert_model(args.model, args.model_directory, device)
        predict_proba = make_kobert_predict_proba_fn(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,)
    else:
        model, vectorizer, ckpt_path = load_ClassicalML_checkpoint(args.model_directory, args.model)
        predict_proba = make_ClassicalML_predict_proba_fn(model, vectorizer, preprocess)

    probs = predict_proba(test_df["processed_text"].tolist())
    test_df["y_pred"] = probs.argmax(axis=1)
    test_df["true_label"] = [class_names[i] for i in test_df["y_true"]]
    test_df["pred_label"] = [class_names[i] for i in test_df["y_pred"]]
    test_df["pred_confidence"] = probs.max(axis=1)

    for class_idx, class_name in enumerate(class_names):
        test_df[f"prob_{class_name}"] = probs[:, class_idx]

    # Save overall prediction results
    pred_csv_path = output_dir / "test_predictions_for_lime.csv"
    test_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")

    acc = accuracy_score(test_df["y_true"], test_df["y_pred"])
    report = classification_report(
        test_df["y_true"],
        test_df["y_pred"],
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(test_df["y_true"], test_df["y_pred"])

    summary = {
        "checkpoint_path": ckpt_path,
        "test_file": os.path.join(args.data_directory, args.test_file),
        "accuracy": float(acc),
        "class_labels_in_encoder_order": class_labels,
        "class_names_in_model_output_order": class_names,
        "confusion_matrix": cm.tolist(),
        "selection": args.selection,
        "num_explain": args.num_explain,
        "lime_num_features": args.lime_num_features,
        "lime_num_samples": args.lime_num_samples,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write("[Classification Report]\n")
        f.write(report)
        f.write("\n\n[Confusion Matrix]\n")
        f.write(str(cm))
        f.write("\n\n[Class order]\n")
        for idx, name in enumerate(class_names):
            f.write(f"{idx}: {name}\n")

    print(f"[INFO] Saved prediction CSV: {pred_csv_path}")
    print(f"[INFO] Accuracy: {acc:.6f}")
    print(report)

    explainer = LimeTextExplainer(
        class_names=class_names,
        split_expression=r"\s+",
        bow=True,
        random_state=args.seed,
    )

    selected_indices = select_indices(test_df, args.selection, args.num_explain, args.seed)
    lime_rows = []

    print(f"[INFO] Explaining {len(selected_indices)} samples with LIME...")

    for rank, idx in enumerate(selected_indices, start=1):
        row = test_df.loc[idx]

        # Explain predicted class. You can also use top_labels=3 if you want all top labels.
        pred_label_idx = int(row["y_pred"])

        exp = explainer.explain_instance(
            text_instance=row["processed_text"],
            classifier_fn=predict_proba,
            labels=[pred_label_idx],
            num_features=args.lime_num_features,
            num_samples=args.lime_num_samples,
        )

        safe_true = str(row["true_label"]).replace("/", "_")
        safe_pred = str(row["pred_label"]).replace("/", "_")
        html_path = html_dir / f"lime_idx_{idx}_true_{safe_true}_pred_{safe_pred}.html"
        exp.save_to_file(str(html_path))

        weights = exp.as_list(label=pred_label_idx)

        for token, weight in weights:
            lime_rows.append({
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

        print(
            f"[{rank}/{len(selected_indices)}] idx={idx}, "
            f"true={row['true_label']}, pred={row['pred_label']}, "
            f"conf={row['pred_confidence']:.4f}, html={html_path}"
        )

    lime_df = pd.DataFrame(lime_rows)
    lime_csv_path = output_dir / "lime_token_weights.csv"
    lime_df.to_csv(lime_csv_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] Saved LIME token weights: {lime_csv_path}")
    print(f"[DONE] Saved LIME HTML files under: {html_dir}")


if __name__ == "__main__":
    main()
