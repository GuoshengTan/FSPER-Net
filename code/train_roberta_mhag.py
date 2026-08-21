"""
Train RoBERTa-BiGRU-Attention for Chinese telecom fraud text classification.

Model:
    Chinese RoBERTa/BERT
      -> BiGRU sequence encoder
      -> additive attention pooling
      -> classifier

The final model is selected from ablation experiments. Extra residual
multi-head attention and gated fusion are kept as optional ablation modes, while
the default model uses RoBERTa + BiGRU + Attention.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
FINAL_MODEL_NAME = "RoBERTa-BiGRU-Attention"
os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_CACHE_DIR / "hub"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


LABEL_NORMALIZATION = {
    "0": "正常文本",
    "冒充客服服务": "冒充电商物流客服类",
    "贷款、代办信用卡类": "贷款、代办信用卡类",
    "冒充公检法及政府机关类": "冒充公检法及政府机关类",
    "冒充领导、熟人类": "冒充领导、熟人类",
}

LABEL_DESCRIPTION_TEMPLATES = {
    "正常文本": "正常文本：不包含诈骗、诱导转账、冒充身份、虚假投资、虚假贷款等风险意图的普通通知、生活交流或服务文本。",
    "正常短信": "正常短信：不包含垃圾营销或诈骗风险的普通短信内容。",
    "垃圾短信": "垃圾短信：包含骚扰营销、诱导点击、虚假推广或其他非正常通信意图的短信内容。",
    "冒充电商物流客服类": "冒充电商物流客服类：诈骗者冒充电商平台、快递物流或客服人员，以退款、理赔、订单异常等理由诱导用户转账或泄露信息。",
    "贷款、代办信用卡类": "贷款、代办信用卡类：诈骗者以贷款、信用卡办理、提额、低息借款等名义诱导用户支付费用、提供验证码或提交敏感信息。",
    "冒充公检法及政府机关类": "冒充公检法及政府机关类：诈骗者冒充公安、检察院、法院或政府机关，以案件、通缉、资金审查等理由威胁并诱导受害者转账。",
    "冒充领导、熟人类": "冒充领导、熟人类：诈骗者冒充领导、亲友或熟人，以临时周转、办事打点、紧急借款等理由要求受害者转账。",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def build_label_descriptions(label_names: Sequence[str]) -> List[str]:
    descriptions = []
    for label in label_names:
        if label in LABEL_DESCRIPTION_TEMPLATES:
            descriptions.append(LABEL_DESCRIPTION_TEMPLATES[label])
        elif "正常" in label:
            descriptions.append(f"{label}：不包含诈骗意图、风险诱导或异常资金操作的普通文本。")
        else:
            descriptions.append(
                f"{label}：该类别表示与{label}相关的电信网络诈骗风险文本，通常包含身份冒充、利益诱导、资金转移或敏感信息获取等欺诈意图。"
            )
    return descriptions


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
            label = str(row.get("label", "")).strip()
            label = LABEL_NORMALIZATION.get(label, label)
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


class FraudTextDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_len: int) -> None:
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class AdditiveAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        energy = torch.tanh(self.proj(hidden))
        scores = self.score(energy).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return pooled, weights


class ResidualMultiHeadAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        padding_mask = ~mask.bool()
        attn_out, _ = self.mha(x, x, x, key_padding_mask=padding_mask, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class GatedSequenceFusion(nn.Module):
    def __init__(self, hidden_dim: int, gru_hidden_dim: int, dropout: float, fusion_type: str) -> None:
        super().__init__()
        self.fusion_type = fusion_type
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_proj = nn.Linear(gru_hidden_dim * 2, hidden_dim)
        if fusion_type == "concat":
            self.concat_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, base_hidden: torch.Tensor, attn_hidden: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.bigru(base_hidden)
        seq_hidden = self.gru_proj(gru_out)
        if self.fusion_type == "gate":
            gate = self.gate(torch.cat([attn_hidden, seq_hidden], dim=-1))
            fused = gate * attn_hidden + (1.0 - gate) * seq_hidden
        elif self.fusion_type == "concat":
            fused = self.concat_proj(torch.cat([attn_hidden, seq_hidden], dim=-1))
        elif self.fusion_type == "sum":
            fused = 0.5 * (attn_hidden + seq_hidden)
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")
        return self.norm(base_hidden + self.dropout(fused))


class RoBertaMHAGClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model: str,
        num_classes: int,
        num_heads: int,
        ffn_dim: int,
        gru_hidden_dim: int,
        dropout: float,
        cache_dir: Path,
        freeze_encoder: bool = False,
        local_files_only: bool = True,
        ablation: str = "full",
        fusion_type: str = "gate",
        use_label_prototypes: bool = False,
        label_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.ablation = ablation
        self.use_label_prototypes = use_label_prototypes
        self.label_temperature = label_temperature
        self.encoder = AutoModel.from_pretrained(pretrained_model, cache_dir=cache_dir, local_files_only=local_files_only)
        hidden_dim = self.encoder.config.hidden_size
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.mhar = ResidualMultiHeadAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout)
        self.fusion = GatedSequenceFusion(hidden_dim, gru_hidden_dim, dropout, fusion_type=fusion_type)
        self.bigru_only = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.bigru_only_proj = nn.Linear(gru_hidden_dim * 2, hidden_dim)
        self.bigru_only_norm = nn.LayerNorm(hidden_dim)
        self.pooling = AdditiveAttentionPooling(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        if use_label_prototypes:
            self.label_prototypes = nn.Parameter(torch.empty(num_classes, hidden_dim))
            nn.init.normal_(self.label_prototypes, mean=0.0, std=0.02)
        else:
            self.register_parameter("label_prototypes", None)

    def label_similarity_logits(self, features: torch.Tensor) -> Optional[torch.Tensor]:
        if self.label_prototypes is None:
            return None
        text_features = F.normalize(features, p=2, dim=-1)
        label_features = F.normalize(self.label_prototypes, p=2, dim=-1)
        return torch.matmul(text_features, label_features.t()) / self.label_temperature

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        base_hidden = outputs.last_hidden_state
        if self.ablation == "roberta_only":
            fused_hidden = base_hidden
        elif self.ablation == "no_mha":
            gru_out, _ = self.bigru_only(base_hidden)
            seq_hidden = self.bigru_only_proj(gru_out)
            fused_hidden = self.bigru_only_norm(base_hidden + self.dropout(seq_hidden))
        elif self.ablation == "no_bigru":
            fused_hidden = self.mhar(base_hidden, attention_mask)
        elif self.ablation in {"full", "no_gate"}:
            attn_hidden = self.mhar(base_hidden, attention_mask)
            fused_hidden = self.fusion(base_hidden, attn_hidden)
        else:
            raise ValueError(f"Unsupported ablation: {self.ablation}")
        pooled, weights = self.pooling(fused_hidden, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        label_logits = self.label_similarity_logits(pooled)
        return {
            "logits": logits,
            "label_logits": label_logits,
            "features": pooled,
            "attention_weights": weights,
        }


class DualRobustLoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor], focal_gamma: float, focal_alpha: float) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, weight=self.class_weights)
        per_sample_ce = F.cross_entropy(logits, labels, weight=self.class_weights, reduction="none")
        pt = torch.exp(-per_sample_ce)
        focal = ((1.0 - pt) ** self.focal_gamma * per_sample_ce).mean()
        return ce + self.focal_alpha * focal


class LSPCLoss(nn.Module):
    def __init__(
        self,
        class_weights: Optional[torch.Tensor],
        focal_gamma: float,
        focal_alpha: float,
        label_loss_weight: float,
        scl_weight: float,
        scl_temperature: float,
    ) -> None:
        super().__init__()
        self.base_loss = DualRobustLoss(class_weights, focal_gamma, focal_alpha)
        self.label_loss_weight = label_loss_weight
        self.scl_weight = scl_weight
        self.scl_temperature = scl_temperature

    def supervised_contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=-1)
        logits = torch.matmul(features, features.t()) / self.scl_temperature
        batch_size = labels.size(0)
        eye = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~eye
        if not positive_mask.any():
            return features.new_tensor(0.0)

        logits = logits.masked_fill(eye, -1e9)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_count = positive_mask.sum(dim=1).clamp_min(1)
        mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_count
        valid_anchor = positive_mask.any(dim=1)
        return -mean_log_prob_pos[valid_anchor].mean()

    def forward(self, outputs: Dict[str, Optional[torch.Tensor]], labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = outputs["logits"]
        assert logits is not None
        loss_cls = self.base_loss(logits, labels)
        loss = loss_cls
        parts = {"loss_cls": float(loss_cls.detach().cpu())}

        label_logits = outputs.get("label_logits")
        if label_logits is not None and self.label_loss_weight > 0:
            loss_label = F.cross_entropy(label_logits, labels)
            loss = loss + self.label_loss_weight * loss_label
            parts["loss_label"] = float(loss_label.detach().cpu())
        else:
            parts["loss_label"] = 0.0

        features = outputs.get("features")
        if features is not None and self.scl_weight > 0:
            loss_scl = self.supervised_contrastive_loss(features, labels)
            loss = loss + self.scl_weight * loss_scl
            parts["loss_scl"] = float(loss_scl.detach().cpu())
        else:
            parts["loss_scl"] = 0.0

        parts["loss_total"] = float(loss.detach().cpu())
        return loss, parts


@torch.no_grad()
def initialize_label_prototypes(
    model: RoBertaMHAGClassifier,
    tokenizer,
    label_descriptions: Sequence[str],
    max_len: int,
    device: torch.device,
) -> None:
    if model.label_prototypes is None:
        return
    was_training = model.training
    model.eval()
    encoded = tokenizer(
        list(label_descriptions),
        max_length=max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    hidden = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    prototypes = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    model.label_prototypes.copy_(F.normalize(prototypes, p=2, dim=-1))
    model.train(was_training)


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


def compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=num_classes)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float)


def make_weighted_sampler(labels: Sequence[int], num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(np.asarray(labels), minlength=num_classes)
    weights = 1.0 / np.maximum(counts, 1)
    sample_weights = [weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def summarize_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    return {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def json_safe_args(args: argparse.Namespace) -> Dict[str, object]:
    safe = {}
    for key, value in vars(args).items():
        safe[key] = str(value) if isinstance(value, Path) else value
    return safe


def current_lrs(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    return {f"group_{idx}": float(group["lr"]) for idx, group in enumerate(optimizer.param_groups)}


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_checkpoint(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Optional[Path], device: torch.device) -> Optional[Dict[str, object]]:
    if path is None:
        return None
    if not path.exists() or path.stat().st_size == 0:
        print(json.dumps({"checkpoint": str(path), "status": "not_found_start_fresh"}, ensure_ascii=False))
        return None
    return torch.load(path, map_location=device, weights_only=False)


def combine_prediction_logits(outputs: Dict[str, Optional[torch.Tensor]], label_logit_weight: float) -> torch.Tensor:
    logits = outputs["logits"]
    assert logits is not None
    label_logits = outputs.get("label_logits")
    if label_logits is not None and label_logit_weight > 0:
        return logits + label_logit_weight * label_logits
    return logits


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip, epoch):
    model.train()
    total_loss = 0.0
    total_parts = Counter()
    for batch in tqdm(loader, desc=f"train {epoch}", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids, attention_mask)
        loss, parts = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * labels.size(0)
        for key, value in parts.items():
            total_parts[key] += value * labels.size(0)
    avg_parts = {key: float(value / len(loader.dataset)) for key, value in total_parts.items()}
    return total_loss / len(loader.dataset), avg_parts


@torch.no_grad()
def evaluate(model, loader, criterion, device, label_logit_weight):
    model.eval()
    total_loss = 0.0
    total_parts = Counter()
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids, attention_mask)
        loss, parts = criterion(outputs, labels)
        pred = combine_prediction_logits(outputs, label_logit_weight).argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        for key, value in parts.items():
            total_parts[key] += value * labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    avg_parts = {key: float(value / len(loader.dataset)) for key, value in total_parts.items()}
    return total_loss / len(loader.dataset), avg_parts, y_true, y_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["telecom5", "fgrc_scd", "spam_message"], default="telecom5")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "Telecom_Fraud_Texts_5")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "roberta_bigru_attention")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--pretrained-model", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=8e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--gru-hidden-dim", type=int, default=128)
    parser.add_argument("--ablation", choices=["full", "roberta_only", "no_mha", "no_bigru", "no_gate"], default="no_mha")
    parser.add_argument("--fusion-type", choices=["gate", "concat", "sum"], default="gate")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.5)
    parser.add_argument("--use-label-prototypes", action="store_true")
    parser.add_argument("--label-loss-weight", type=float, default=0.0)
    parser.add_argument("--label-logit-weight", type=float, default=0.0)
    parser.add_argument("--label-temperature", type=float, default=0.07)
    parser.add_argument("--scl-weight", type=float, default=0.0)
    parser.add_argument("--scl-temperature", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--max-samples-per-class", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-best-by", choices=["valid_macro_f1", "valid_accuracy", "valid_weighted_f1"], default="valid_macro_f1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    groups = None
    if args.dataset == "telecom5":
        texts, raw_labels = load_telecom_fraud_texts_5(args.data_dir)
    elif args.dataset == "fgrc_scd":
        texts, raw_labels, groups = load_fgrc_scd(args.data_dir)
    elif args.dataset == "spam_message":
        texts, raw_labels = load_spam_message(args.data_dir)
    else:
        raise ValueError(args.dataset)
    texts, raw_labels, groups = cap_per_class(
        texts, raw_labels, groups, args.max_samples_per_class, args.seed
    )
    if not texts:
        raise RuntimeError(f"No data loaded from {args.data_dir}")
    label_names = sorted(set(raw_labels))
    label_descriptions = build_label_descriptions(label_names)
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    labels = [label_to_id[label] for label in raw_labels]
    if groups is None:
        train_texts, valid_texts, test_texts, train_labels, valid_labels, test_labels = make_splits(texts, labels, args.seed)
    else:
        train_texts, valid_texts, test_texts, train_labels, valid_labels, test_labels = make_group_splits(
            texts, labels, groups, args.seed
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_download,
    )
    train_ds = FraudTextDataset(train_texts, train_labels, tokenizer, args.max_len)
    valid_ds = FraudTextDataset(valid_texts, valid_labels, tokenizer, args.max_len)
    test_ds = FraudTextDataset(test_texts, test_labels, tokenizer, args.max_len)

    sampler = make_weighted_sampler(train_labels, len(label_names)) if args.weighted_sampler else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoBertaMHAGClassifier(
        pretrained_model=args.pretrained_model,
        num_classes=len(label_names),
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        dropout=args.dropout,
        cache_dir=args.cache_dir,
        freeze_encoder=args.freeze_encoder,
        local_files_only=not args.allow_download,
        ablation=args.ablation,
        fusion_type=("concat" if args.ablation == "no_gate" else args.fusion_type),
        use_label_prototypes=args.use_label_prototypes,
        label_temperature=args.label_temperature,
    ).to(device)
    initialize_label_prototypes(model, tokenizer, label_descriptions, args.max_len, device)
    param_info = count_parameters(model)

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append((name, param))
        else:
            head_params.append((name, param))
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in encoder_params if not any(nd in n for nd in no_decay)], "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": [p for n, p in encoder_params if any(nd in n for nd in no_decay)], "lr": args.lr, "weight_decay": 0.0},
            {"params": [p for _, p in head_params], "lr": args.head_lr, "weight_decay": args.weight_decay},
        ]
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    weights = None if args.no_class_weight else compute_class_weights(train_labels, len(label_names)).to(device)
    criterion = LSPCLoss(
        weights,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        label_loss_weight=args.label_loss_weight,
        scl_weight=args.scl_weight,
        scl_temperature=args.scl_temperature,
    )

    history = []
    best_score = -1.0
    start_epoch = 1
    state = load_checkpoint(args.resume, device)
    if state:
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        history = list(state.get("history", []))
        best_score = float(state.get("best_score", -1.0))
        start_epoch = int(state["epoch"]) + 1
        print(json.dumps({"resume_from": str(args.resume), "next_epoch": start_epoch, "best_score": best_score}, ensure_ascii=False))

    save_json(args.output_dir / "label_mapping.json", {"label_to_id": label_to_id, "id_to_label": id_to_label})
    save_json(
        args.output_dir / "dataset_summary.json",
        {
            "total": len(texts),
            "dataset": args.dataset,
            "group_split": groups is not None,
            "train": len(train_ds),
            "valid": len(valid_ds),
            "test": len(test_ds),
            "labels": Counter(raw_labels),
            "pretrained_model": args.pretrained_model,
            "model_name": FINAL_MODEL_NAME,
            "ablation": args.ablation,
            "fusion_type": "concat" if args.ablation == "no_gate" else args.fusion_type,
            "use_label_prototypes": args.use_label_prototypes,
            "max_len": args.max_len,
            **param_info,
        },
    )
    experiment_config = {
        "model_name": FINAL_MODEL_NAME,
        "created_at_unix": time.time(),
        "args": json_safe_args(args),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "label_descriptions": {label: desc for label, desc in zip(label_names, label_descriptions)},
        "label_distribution": Counter(raw_labels),
        "split_sizes": {
            "train": len(train_ds),
            "valid": len(valid_ds),
            "test": len(test_ds),
        },
        "parameter_info": param_info,
        "optimizer": {
            "name": "AdamW",
            "configured_encoder_lr": args.lr,
            "configured_head_lr": args.head_lr,
            "initial_lrs": current_lrs(optimizer),
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
        },
        "criterion": {
            "name": "CrossEntropy + focal + optional label semantic loss + optional supervised contrastive loss",
            "class_weight": not args.no_class_weight,
            "weighted_sampler": args.weighted_sampler,
            "focal_gamma": args.focal_gamma,
            "focal_alpha": args.focal_alpha,
            "use_label_prototypes": args.use_label_prototypes,
            "label_loss_weight": args.label_loss_weight,
            "label_logit_weight": args.label_logit_weight,
            "label_temperature": args.label_temperature,
            "scl_weight": args.scl_weight,
            "scl_temperature": args.scl_temperature,
        },
        "selection": {
            "save_best_by": args.save_best_by,
        },
    }
    save_json(args.output_dir / "experiment_config.json", experiment_config)

    def checkpoint_payload(epoch: int):
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "history": history,
            "args": json_safe_args(args),
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "label_descriptions": {label: desc for label, desc in zip(label_names, label_descriptions)},
            "parameter_info": param_info,
            "saved_at_unix": time.time(),
        }

    print(json.dumps({"device": str(device), **param_info}, ensure_ascii=False))
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_loss_parts = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, args.grad_clip, epoch)
        valid_loss, valid_loss_parts, valid_true, valid_pred = evaluate(
            model, valid_loader, criterion, device, args.label_logit_weight
        )
        valid_metrics = summarize_metrics(valid_true, valid_pred)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_parts": train_loss_parts,
            "valid_loss": valid_loss,
            "valid_loss_parts": valid_loss_parts,
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_precision": valid_metrics["macro_precision"],
            "valid_macro_recall": valid_metrics["macro_recall"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_weighted_precision": valid_metrics["weighted_precision"],
            "valid_weighted_recall": valid_metrics["weighted_recall"],
            "valid_weighted_f1": valid_metrics["weighted_f1"],
            "learning_rates": current_lrs(optimizer),
            "saved_at_unix": time.time(),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        score = row[args.save_best_by]
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": json_safe_args(args),
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                    "label_descriptions": {label: desc for label, desc in zip(label_names, label_descriptions)},
                    "best_score": best_score,
                    "best_epoch": epoch,
                    "parameter_info": param_info,
                },
                args.output_dir / "best_model.pt",
            )
            save_json(
                args.output_dir / "best_metrics.json",
                {
                    "best_epoch": epoch,
                    "best_score": best_score,
                    "save_best_by": args.save_best_by,
                    "validation": row,
                    "args": json_safe_args(args),
                    "parameter_info": param_info,
                    "saved_at_unix": time.time(),
                },
            )

        save_checkpoint(args.output_dir / "latest_checkpoint.pt", checkpoint_payload(epoch))
        save_json(args.output_dir / "training_history.json", history)

    best = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_loss, test_loss_parts, test_true, test_pred = evaluate(model, test_loader, criterion, device, args.label_logit_weight)
    target_names = [id_to_label[i] for i in range(len(label_names))]
    report = classification_report(test_true, test_pred, target_names=target_names, digits=4, output_dict=True, zero_division=0)
    result = {
        "best_epoch": best.get("best_epoch"),
        "best_score": best.get("best_score"),
        "save_best_by": args.save_best_by,
        "test_loss": test_loss,
        "test_loss_parts": test_loss_parts,
        "test_accuracy": float(accuracy_score(test_true, test_pred)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(test_true, test_pred).tolist(),
        "test_true": test_true,
        "test_pred": test_pred,
        "labels": target_names,
        "history": history,
    }
    save_json(args.output_dir / "metrics.json", result)
    macro_avg = report.get("macro avg", {})
    weighted_avg = report.get("weighted avg", {})
    final_summary = {
        "model_name": FINAL_MODEL_NAME,
        "output_dir": str(args.output_dir),
        "best_epoch": best.get("best_epoch"),
        "best_score": best.get("best_score"),
        "save_best_by": args.save_best_by,
        "test_loss": test_loss,
        "test_loss_parts": test_loss_parts,
        "test_accuracy": float(accuracy_score(test_true, test_pred)),
        "test_macro_precision": macro_avg.get("precision"),
        "test_macro_recall": macro_avg.get("recall"),
        "test_macro_f1": macro_avg.get("f1-score"),
        "test_weighted_precision": weighted_avg.get("precision"),
        "test_weighted_recall": weighted_avg.get("recall"),
        "test_weighted_f1": weighted_avg.get("f1-score"),
        "args": json_safe_args(args),
        "parameter_info": param_info,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "label_descriptions": {label: desc for label, desc in zip(label_names, label_descriptions)},
        "saved_at_unix": time.time(),
    }
    save_json(args.output_dir / "final_summary.json", final_summary)
    print(json.dumps({"best_epoch": result["best_epoch"], "test_accuracy": result["test_accuracy"], "test_loss": test_loss}, ensure_ascii=False, indent=2))
    print(classification_report(test_true, test_pred, target_names=target_names, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
