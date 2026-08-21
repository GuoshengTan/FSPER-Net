"""
Train classical and neural baselines on the three manuscript datasets.

Supported models:
    svm, logistic_regression, random_forest,
    textcnn, bigru, bigru_attention
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from split_manifest import split_from_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL_NORMALIZATION = {
    "0": "正常文本",
    "冒充客服服务": "冒充电商物流客服类",
    "贷款、代办信用卡类": "贷款、代办信用卡类",
    "冒充公检法及政府机关类": "冒充公检法及政府机关类",
    "冒充领导、熟人类": "冒充领导、熟人类",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def read_csv_with_fallback(path: Path) -> List[Dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {path}")


def load_telecom_fraud_texts_5(data_dir: Path) -> Tuple[List[str], List[str]]:
    texts, labels, seen = [], [], set()
    for path in sorted(data_dir.glob("label*-last.csv")):
        for row in read_csv_with_fallback(path):
            text = normalize_text(row.get("content", ""))
            label = LABEL_NORMALIZATION.get(str(row.get("label", "")).strip(), str(row.get("label", "")).strip())
            if not text or not label:
                continue
            key = (text, label)
            if key in seen:
                continue
            seen.add(key)
            texts.append(text)
            labels.append(label)
    return texts, labels


def load_fgrc_scd(data_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    path = data_dir / "finetuning_initial.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    texts, labels, groups, seen = [], [], [], set()
    for row in rows:
        text = normalize_text(row.get("文本", ""))
        label = str(row.get("风险类别", "")).strip()
        group = str(row.get("案件编号", "")).strip()
        if not text or not label or not group:
            continue
        key = (text, label)
        if key in seen:
            continue
        seen.add(key)
        texts.append(text)
        labels.append(label)
        groups.append(group)
    return texts, labels, groups


def load_spam_message(data_dir: Path) -> Tuple[List[str], List[str]]:
    path = data_dir / "带标签短信.txt"
    texts, labels, seen = [], [], set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            raw_label, text = line.split("\t", 1)
            text = normalize_text(text)
            label = {"0": "正常短信", "1": "垃圾短信"}.get(raw_label.strip())
            if not text or label is None:
                continue
            key = (text, label)
            if key in seen:
                continue
            seen.add(key)
            texts.append(text)
            labels.append(label)
    return texts, labels


def load_dataset(dataset: str, data_dir: Path) -> Tuple[List[str], List[str], Optional[List[str]]]:
    if dataset == "telecom5":
        texts, labels = load_telecom_fraud_texts_5(data_dir)
        return texts, labels, None
    if dataset == "fgrc_scd":
        return load_fgrc_scd(data_dir)
    if dataset == "spam_message":
        texts, labels = load_spam_message(data_dir)
        return texts, labels, None
    raise ValueError(dataset)


def cap_per_class(
    texts: List[str],
    labels: List[str],
    groups: Optional[List[str]],
    max_samples_per_class: int,
    seed: int,
) -> Tuple[List[str], List[str], Optional[List[str]]]:
    if max_samples_per_class <= 0:
        return texts, labels, groups
    rng = random.Random(seed)
    by_label: Dict[str, List[int]] = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(label, []).append(idx)
    selected = []
    for indices in by_label.values():
        rng.shuffle(indices)
        selected.extend(indices[:max_samples_per_class])
    selected.sort()
    capped_groups = [groups[i] for i in selected] if groups is not None else None
    return [texts[i] for i in selected], [labels[i] for i in selected], capped_groups


def make_splits(texts: List[str], labels: List[int], seed: int):
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    valid_texts, test_texts, valid_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=seed, stratify=temp_labels
    )
    return train_texts, valid_texts, test_texts, train_labels, valid_labels, test_labels


def make_group_splits(texts: List[str], labels: List[int], groups: List[str], seed: int):
    indices = np.arange(len(texts))
    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, temp_idx = next(first.split(indices, labels, groups))
    temp_groups = np.asarray(groups, dtype=object)[temp_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    valid_rel, test_rel = next(second.split(temp_idx, np.asarray(labels)[temp_idx], temp_groups))
    valid_idx, test_idx = temp_idx[valid_rel], temp_idx[test_rel]

    def take(values, idx):
        return [values[i] for i in idx]

    return (
        take(texts, train_idx),
        take(texts, valid_idx),
        take(texts, test_idx),
        take(labels, train_idx),
        take(labels, valid_idx),
        take(labels, test_idx),
    )


def char_tokenize(text: str) -> List[str]:
    return list(text)


def build_vocab(texts: Sequence[str], max_vocab_size: int, min_freq: int) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(char_tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common(max_vocab_size - len(vocab)):
        if freq < min_freq:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(token, vocab["<unk>"]) for token in char_tokenize(text)[:max_len]]
    if len(ids) < max_len:
        ids.extend([vocab["<pad>"]] * (max_len - len(ids)))
    return ids


class CharDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: Dict[str, int], max_len: int) -> None:
        self.ids = [encode_text(text, vocab, max_len) for text in texts]
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_ids": torch.tensor(self.ids[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int, channels: int, kernels: Sequence[int], dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * len(kernels), num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids).transpose(1, 2)
        feats = []
        for conv in self.convs:
            h = F.relu(conv(x))
            feats.append(F.max_pool1d(h, kernel_size=h.size(-1)).squeeze(-1))
        return self.classifier(self.dropout(torch.cat(feats, dim=1)))


class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int, hidden_dim: int, dropout: float, attention: bool):
        super().__init__()
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bigru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.attn_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attn_score = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        mask = input_ids.ne(0)
        x = self.embedding(input_ids)
        h, _ = self.bigru(x)
        if self.attention:
            score = self.attn_score(torch.tanh(self.attn_proj(h))).squeeze(-1)
            score = score.masked_fill(~mask, -1e9)
            weight = torch.softmax(score, dim=1)
            pooled = torch.bmm(weight.unsqueeze(1), h).squeeze(1)
        else:
            lengths = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
            pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / lengths
        return self.classifier(self.dropout(pooled))


def summarize_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    return {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def train_sklearn(args, train_texts, valid_texts, test_texts, train_labels, valid_labels, test_labels, label_names):
    if args.model == "svm":
        clf = LinearSVC(class_weight="balanced", random_state=args.seed)
    elif args.model == "logistic_regression":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
    elif args.model == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=args.seed, n_jobs=-1)
    else:
        raise ValueError(args.model)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=args.max_features)),
            ("clf", clf),
        ]
    )
    pipeline.fit(train_texts, train_labels)
    valid_pred = pipeline.predict(valid_texts)
    test_pred = pipeline.predict(test_texts)
    valid_metrics = summarize_metrics(valid_labels, valid_pred)
    test_metrics = summarize_metrics(test_labels, test_pred)
    label_ids = list(range(len(label_names)))
    report = classification_report(
        test_labels,
        test_pred,
        labels=label_ids,
        target_names=label_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    joblib.dump(pipeline, args.output_dir / "best_model.joblib")
    return {
        "best_epoch": None,
        "best_score": valid_metrics["macro_f1"],
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(test_labels, test_pred, labels=label_ids).tolist(),
        "parameter_info": {"total_parameters": None, "trainable_parameters": None},
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_torch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / len(loader.dataset), y_true, y_pred


def train_torch(args, train_texts, valid_texts, test_texts, train_labels, valid_labels, test_labels, label_names):
    vocab = build_vocab(train_texts, args.max_vocab_size, args.min_freq)
    train_ds = CharDataset(train_texts, train_labels, vocab, args.max_len)
    valid_ds = CharDataset(valid_texts, valid_labels, vocab, args.max_len)
    test_ds = CharDataset(test_texts, test_labels, vocab, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "textcnn":
        model = TextCNN(len(vocab), len(label_names), args.embedding_dim, args.cnn_channels, [2, 3, 4, 5], args.dropout)
    elif args.model == "bigru":
        model = BiGRUClassifier(len(vocab), len(label_names), args.embedding_dim, args.gru_hidden_dim, args.dropout, attention=False)
    elif args.model == "bigru_attention":
        model = BiGRUClassifier(len(vocab), len(label_names), args.embedding_dim, args.gru_hidden_dim, args.dropout, attention=True)
    else:
        raise ValueError(args.model)
    model.to(device)
    weights = np.bincount(np.asarray(train_labels), minlength=len(label_names))
    weights = len(train_labels) / (len(label_names) * np.maximum(weights, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    param_info = count_parameters(model)
    print(json.dumps({"device": str(device), **param_info}, ensure_ascii=False))
    history = []
    best_score = -1.0
    best_state = None
    best_epoch = 0
    start_epoch = 1

    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = list(state.get("history", []))
        best_score = float(state.get("best_score", -1.0))
        best_epoch = int(state.get("best_epoch", 0))
        best_state = state.get("best_state_dict")
        start_epoch = int(state["epoch"]) + 1
        rng_state = state.get("rng_state")
        if rng_state:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
        print(
            json.dumps(
                {
                    "resume_from": str(args.resume),
                    "next_epoch": start_epoch,
                    "best_score": best_score,
                },
                ensure_ascii=False,
            )
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_true, valid_pred = evaluate_torch(model, valid_loader, criterion, device)
        valid_metrics = summarize_metrics(valid_true, valid_pred)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_weighted_f1": valid_metrics["weighted_f1"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if valid_metrics["macro_f1"] > best_score:
            best_score = valid_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_state_dict": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "history": history,
                "vocab": vocab,
                "args": vars(args),
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            },
            args.output_dir / "latest_checkpoint.pt",
        )
        save_json(args.output_dir / "training_history.json", history)

    if best_state is None:
        raise RuntimeError("No best model state is available; the checkpoint may be incomplete.")
    model.load_state_dict(best_state)
    test_loss, test_true, test_pred = evaluate_torch(model, test_loader, criterion, device)
    test_metrics = summarize_metrics(test_true, test_pred)
    label_ids = list(range(len(label_names)))
    report = classification_report(
        test_true,
        test_pred,
        labels=label_ids,
        target_names=label_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    torch.save({"model_state_dict": best_state, "vocab": vocab, "args": vars(args)}, args.output_dir / "best_model.pt")
    save_json(args.output_dir / "training_history.json", history)
    return {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "valid_metrics": history[best_epoch - 1],
        "test_metrics": test_metrics,
        "test_loss": test_loss,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(test_true, test_pred, labels=label_ids).tolist(),
        "parameter_info": param_info,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["telecom5", "fgrc_scd", "spam_message"], default="telecom5")
    parser.add_argument("--model", choices=["svm", "logistic_regression", "random_forest", "textcnn", "bigru", "bigru_attention"], required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "Telecom_Fraud_Texts_5")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--max-vocab-size", type=int, default=6000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--cnn-channels", type=int, default=128)
    parser.add_argument("--gru-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-samples-per-class", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional validated train/valid/test index manifest.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    texts, raw_labels, groups = load_dataset(args.dataset, args.data_dir)
    texts, raw_labels, groups = cap_per_class(
        texts,
        raw_labels,
        groups,
        args.max_samples_per_class,
        args.seed,
    )
    if not texts:
        raise RuntimeError(f"No data loaded from {args.data_dir}")
    label_names = sorted(set(raw_labels))
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    labels = [label_to_id[label] for label in raw_labels]
    split_manifest = None
    if args.split_manifest is not None:
        split, split_manifest = split_from_manifest(
            args.split_manifest,
            texts,
            labels,
            groups,
        )
        (
            train_texts,
            valid_texts,
            test_texts,
            train_labels,
            valid_labels,
            test_labels,
        ) = split
    elif groups is None:
        train_texts, valid_texts, test_texts, train_labels, valid_labels, test_labels = make_splits(
            texts,
            labels,
            args.seed,
        )
    else:
        train_texts, valid_texts, test_texts, train_labels, valid_labels, test_labels = make_group_splits(
            texts,
            labels,
            groups,
            args.seed,
        )

    def label_counts(values: Sequence[int]) -> Dict[str, int]:
        counts = Counter(values)
        return {label_names[idx]: int(counts.get(idx, 0)) for idx in range(len(label_names))}

    save_json(
        args.output_dir / "dataset_summary.json",
        {
            "dataset": args.dataset,
            "total_samples": len(texts),
            "num_classes": len(label_names),
            "group_split": groups is not None,
            "split_manifest": (
                str(args.split_manifest.resolve())
                if args.split_manifest is not None
                else None
            ),
            "test_role": (
                split_manifest.get("test_role")
                if split_manifest is not None
                else "generated_test"
            ),
            "train_samples": len(train_texts),
            "valid_samples": len(valid_texts),
            "test_samples": len(test_texts),
            "train_label_counts": label_counts(train_labels),
            "valid_label_counts": label_counts(valid_labels),
            "test_label_counts": label_counts(test_labels),
        },
    )

    if args.model in {"svm", "logistic_regression", "random_forest"}:
        result = train_sklearn(args, train_texts, valid_texts, test_texts, train_labels, valid_labels, test_labels, label_names)
    else:
        result = train_torch(args, train_texts, valid_texts, test_texts, train_labels, valid_labels, test_labels, label_names)

    test_metrics = result["test_metrics"]
    summary = {
        "dataset": args.dataset,
        "model_name": args.model,
        "best_epoch": result["best_epoch"],
        "best_score": result["best_score"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_precision": test_metrics["weighted_precision"],
        "test_weighted_recall": test_metrics["weighted_recall"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "parameter_info": result["parameter_info"],
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "saved_at_unix": time.time(),
    }
    metrics = {
        "best_epoch": result["best_epoch"],
        "best_score": result["best_score"],
        "classification_report": result["classification_report"],
        "confusion_matrix": result["confusion_matrix"],
        "labels": label_names,
    }
    save_json(args.output_dir / "final_summary.json", summary)
    save_json(args.output_dir / "metrics.json", metrics)
    save_json(args.output_dir / "experiment_config.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
